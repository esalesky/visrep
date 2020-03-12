""" Train a sentence embedding model."""
import os
import argparse
import sys
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision.utils import save_image
from torchsummary import summary
from augment import OcrAug
from ocr_dataset import LineSynthDataset, LmdbDataset, ImageGroupSampler, image_collater
from tqdm import tqdm
import torch.nn.functional as F
import logging
from text_utils import compute_cer_wer, uxxxx_to_utf8, utf8_to_uxxxx

from model import OcrDecoder, OcrEncoder, OcrModel

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', type=str,
                        default='', help='Input train seed text or lmdb')
    parser.add_argument('--test-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument('--test-background', type=str,
                        default='', help='Input train background file')
    parser.add_argument("--test-max-image-width", type=int,
                        default=1500, help="max train image width")
    parser.add_argument("--test-min-image-width", type=int,
                        default=32, help="min train image width")
    parser.add_argument('--test-use-default-image', action='store_true',
                        help='Use default image ')
    parser.add_argument("--test-max-seed", type=int,
                        default=500000, help="max seed")

    parser.add_argument("--image-height", type=int,
                        default=32, help="Image height")

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument("--batch-size", type=int,
                        default=8, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="Nbr dataset workers")

    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="Encoder dim")
    parser.add_argument("--encoder-arch", type=str,
                        default='vista', help="Encoder arch: vista, vista_maxpool, resnet18")

    parser.add_argument("--decoder-lstm-layers", type=int,
                        default=3, help="Number of LSTM layers in model")
    parser.add_argument("--decoder-lstm-units", type=int,
                        default=640, help="Number of LSTM hidden units ")  # 640 256
    parser.add_argument("--decoder-lstm-dropout", type=float,
                        default=0.50, help="Number of LSTM layers in model")

    parser.add_argument('--image-verbose', action='store_true',
                        help='Debug info')

    parser.add_argument("--load-checkpoint-path", type=str,
                        default=None, help="Input checkpoint path")

    parser.add_argument('--write-images', action='store_true',
                        help='write images')

    args = parser.parse_args(argv)
    for arg in vars(args):
        LOG.info('%s %s', arg, getattr(args, arg))

    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def main(args):
    start_time = time.clock()

    LOG.info('__Python VERSION: %s', sys.version)
    LOG.info('__PyTorch VERSION: %s', torch.__version__)
    LOG.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    LOG.info('__Number CUDA Devices: %s', torch.cuda.device_count())
    LOG.info('__Active CUDA Device: GPU %s', torch.cuda.current_device())
    LOG.info('__CUDA_VISIBLE_DEVICES %s ',
             str(os.environ["CUDA_VISIBLE_DEVICES"]))

    decode_output = args.output + '/decode'
    if not os.path.exists(decode_output):
        os.makedirs(decode_output)

    embeddings_output = args.output + '/embeddings'
    if not os.path.exists(embeddings_output):
        os.makedirs(embeddings_output)

    images_output = args.output + '/images'
    if not os.path.exists(images_output):
        os.makedirs(images_output)

    hyp_file = open(os.path.join(decode_output, "hyp-chars.txt"), "w")
    hyp_file_utf8 = open(os.path.join(
        decode_output, "hyp-chars.txt.utf8"), "w")

    checkpoint = torch.load(args.load_checkpoint_path)
    LOG.info('Loading checkpoint...')
    LOG.info(' epoch %d', checkpoint['epoch'])
    LOG.info(' loss %f', checkpoint['loss'])
    LOG.info(' cer %f', checkpoint['cer'])
    LOG.info(' wer %f', checkpoint['wer'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = OcrEncoder(args)
    decoder = OcrDecoder(args, checkpoint['alphabet'])
    model = OcrModel(args, encoder, decoder, checkpoint['alphabet'])
    model.to(device)

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    LOG.info('alphabet %s', (model.dictionary.len()))

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    test_dataset = LineSynthDataset(text_file_path=args.test,
                                    font_file_path=args.test_font,
                                    bkg_file_path=args.test_background,
                                    max_image_width=args.test_max_image_width,
                                    min_image_width=args.test_min_image_width,
                                    image_height=args.image_height,
                                    transform=test_transform,
                                    max_seed=args.test_max_seed,
                                    use_default_image=args.test_use_default_image,
                                    image_verbose=args.image_verbose)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                              sampler=ImageGroupSampler(
                                                  test_dataset, rand=False),
                                              collate_fn=lambda b: image_collater(
                                                  b, args.image_verbose),
                                              num_workers=0, pin_memory=False,
                                              drop_last=False)

    test_dict = test_dataset.alphabet
    print('...Dictionary test', len(test_dict))
    for i in range(len(test_dict)):
        if i < 5:
            print(i, test_dict.idx_to_char[i])
        elif i > len(test_dict) - 5:
            print(i, test_dict.idx_to_char[i])

    total_images = 0
    cer_running_avg = 0
    wer_running_avg = 0
    n_samples = 0
    model.eval()
    print('...Images to process', len(test_loader.dataset))
    with torch.no_grad():
        t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
        for sample_ctr, sample in enumerate(t):

            sample = move_to_cuda(sample)
            batch_size = len(sample['net_input']['src_widths'])
            total_images += batch_size

            net_meta = model(**sample['net_input'])

            hyp_transcriptions = model.decode(
                net_meta['prob_output_view'], net_meta['lstm_output_lengths'])

            batch_cer = 0
            batch_wer = 0
            n_samples += 1

            image_list = sample['net_input']['src_tokens'].cpu().numpy()
            embedding_list = net_meta['lstm_output'].cpu().numpy()
            embedding_list_view = embedding_list.transpose(1, 0, 2)

            decode_metadata = {}
            for hyp_ctr in range(len(hyp_transcriptions)):
                uxxxx_hyp_text = hyp_transcriptions[hyp_ctr]
                utf8_hyp_text = uxxxx_to_utf8(uxxxx_hyp_text)

                utf8_ref_text = sample['seed_text'][hyp_ctr]
                uxxxx_ref_text = utf8_to_uxxxx(utf8_ref_text)

                image_id = sample['image_id'][hyp_ctr]

                print("%s (%s)" % (uxxxx_hyp_text, image_id), file=hyp_file)
                print("%s (%s)" % (utf8_hyp_text, image_id), file=hyp_file_utf8)

                decode_metadata['image_id'] = image_id
                decode_metadata['utf8_ref_text'] = utf8_ref_text
                decode_metadata['uxxxx_ref_text'] = uxxxx_ref_text
                decode_metadata['image'] = image_list[hyp_ctr]
                decode_metadata['embedding'] = embedding_list_view[hyp_ctr]

                cer, wer = compute_cer_wer(
                    uxxxx_hyp_text, uxxxx_ref_text)

                batch_cer += cer
                batch_wer += wer

                # if sample_ctr % 50 == 0:
                #    if hyp_ctr < 2:
                #        print('\nhyp', uxxxx_hyp_text)
                #        print('ref', uxxxx_ref_text)
                #        print('cer %f \twer %f' % (cer, wer))

                if args.write_images:
                    image = np.uint8(
                        image_list[hyp_ctr].transpose((1, 2, 0)) * 255)
                    image = image[:, :, ::-1].copy()  # rgb to bgr
                    cv2.imwrite(os.path.join(images_output,
                                             utf8_ref_text + '.png'), image)

                np.savez_compressed(os.path.join(embeddings_output, str(image_id) + '.npz'), allow_pickle=True,
                                    metadata=decode_metadata)

            cer_running_avg += (batch_cer / batch_size -
                                cer_running_avg) / n_samples
            wer_running_avg += (batch_wer / batch_size -
                                wer_running_avg) / n_samples

    hyp_file.close()
    hyp_file_utf8.close()

    print('total processed', total_images)
    LOG.info('...complete, time %s', time.clock() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
