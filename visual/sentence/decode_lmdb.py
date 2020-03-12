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
    parser.add_argument("--test-split", type=str,
                        default='train', help="train lmdb split")

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument("--image-height", type=int,
                        default=32, help="Image height")

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

    hyp_file = open(os.path.join(decode_output, "hyp-chars.txt"), "w")
    hyp_file_utf8 = open(os.path.join(
        decode_output, "hyp-chars.txt.utf8"), "w")

    checkpoint = torch.load(args.load_checkpoint_path)
    LOG.info('Loading checkpoint...')
    LOG.info(' epoch %d', checkpoint['epoch'])
    LOG.info(' loss %f', checkpoint['loss'])
    LOG.info(' cer %f', checkpoint['cer'])
    LOG.info(' wer %f', checkpoint['wer'])

    encoder = OcrEncoder(args)
    decoder = OcrDecoder(args, checkpoint['alphabet'])
    model = OcrModel(args, encoder, decoder, checkpoint['alphabet'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    test_dataset = LmdbDataset(args.test, args.test_split,
                               test_transform,
                               image_height=args.image_height,
                               image_verbose=args.image_verbose,
                               alphabet=model.alphabet,
                               max_image_width=args.test_max_image_width)
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
    with torch.no_grad():
        for sample_ctr, sample in enumerate(test_loader):

            sample = move_to_cuda(sample)
            batch_size = len(sample['net_input']['src_widths'])
            total_images += batch_size

            total_val_images += batch_size
            net_meta = model(**sample['net_input'])
            log_probs = F.log_softmax(net_meta['prob_output_view'], dim=2)

            hyp_transcriptions = model.decode(
                log_probs, net_meta['lstm_output_lengths'])

            batch_cer = 0
            batch_wer = 0
            n_samples += 1

            for hyp_ctr in range(len(hyp_transcriptions)):
                uxxxx_hyp_text = hyp_transcriptions[hyp_ctr]
                utf8_hyp_text = uxxxx_to_utf8(uxxxx_hyp_text)

                image_id = sample['image_id'][hyp_ctr]

                print("%s (%s)" % (uxxxx_hyp_text, image_id), file=hyp_file)
                print("%s (%s)" % (utf8_hyp_text, image_id), file=hyp_file_utf8)

                utf8_ref_text = sample['seed_text'][hyp_ctr]
                uxxxx_ref_text = utf8_to_uxxxx(utf8_ref_text)

                cer, wer = compute_cer_wer(
                    uxxxx_hyp_text, uxxxx_ref_text)

                batch_cer += cer
                batch_wer += wer

                if sample_ctr % 50 == 0:
                    if hyp_ctr < 2:
                        print('\nhyp', uxxxx_hyp_text)
                        print('ref', uxxxx_ref_text)
                        print('cer %f \twer %f' % (cer, wer))

            cer_running_avg += (batch_cer / batch_size -
                                cer_running_avg) / n_samples
            wer_running_avg += (batch_wer / batch_size -
                                wer_running_avg) / n_samples

    hyp_file.close()
    hyp_file_utf8.close()

    LOG.info('...complete, time %s', time.clock() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
