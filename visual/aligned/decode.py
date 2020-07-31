""" Decode a pre-train visual embedding model."""
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
from torchsummary import summary
from vis_align_dataset import ImageSynthDataset, ImageGroupSampler, image_collater
from tqdm import tqdm
import torch.nn.functional as F
import logging
import unicodedata

from fairseq.data import Dictionary
from vis_align_ocr import AlignOcrModel

import logging
LOG = logging.getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict', type=str,
                        help='Input dictionary')

    parser.add_argument('--test', type=str,
                        default='', help='Input text')
    parser.add_argument('--test-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument("--test-max-text-width", type=int,
                        default=6000, help="max text width")
    parser.add_argument("--test-min-text-width", type=int,
                        default=1, help="max text width")

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument("--image-height", type=int,
                        default=32, help="Image height")
    parser.add_argument("--image-width", type=int,
                        default=32, help="Image width")
    parser.add_argument("--font-size", type=int,
                        default=16, help="Font size")

    parser.add_argument("--batch-size", type=int,
                        default=8, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="Nbr dataset workers")

    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="encoder dimension")

    parser.add_argument("--load-checkpoint-path", type=str,
                        default=None, help="Input checkpoint path")

    parser.add_argument("--test-display-mod", type=int,
                        default=10000, help="test display mod")

    parser.add_argument('--write-image-samples', action='store_true',
                        help='write image samples')
    parser.add_argument('--write-metadata', action='store_true',
                        help='write metadata')

    parser.add_argument('--image-verbose', action='store_true',
                        help='more debug info')

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


def get_datasets(args):

    alphabet = Dictionary.load(args.dict)

    for idx, char in enumerate(alphabet.symbols):
        if idx < 10 or idx > len(alphabet.symbols) - 10 - 1:
            LOG.info('...indicies %d, symbol %s, count %d', alphabet.indices[alphabet.symbols[idx]],
                     alphabet.symbols[idx], alphabet.count[idx])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = ImageSynthDataset(
        text_file_path=args.test,
        font_file=args.test_font,
        transform=test_transform,
        alphabet=alphabet,
        max_text_width=args.test_max_text_width,
        min_text_width=args.test_min_text_width,
        image_height=args.image_height,
        image_width=args.image_width,
        image_cache=None)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              sampler=ImageGroupSampler(
                                                  test_dataset, rand=True),
                                              collate_fn=lambda b: image_collater(
                                                  b),
                                              num_workers=args.num_workers, pin_memory=True,
                                              drop_last=True)

    return test_dataset, test_loader


def main(args):

    if args.image_verbose:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    start_time = time.process_time()

    LOG.info('__Python VERSION: %s', sys.version)
    LOG.info('__PyTorch VERSION: %s', torch.__version__)
    LOG.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    LOG.info('__Number CUDA Devices: %s', torch.cuda.device_count())
    LOG.info('__Active CUDA Device: GPU %s', torch.cuda.current_device())
    LOG.info('__CUDA_VISIBLE_DEVICES %s ',
             str(os.environ["CUDA_VISIBLE_DEVICES"]))

    if args.write_metadata:
        embeddings_images_output = args.output + '/embeddings/images'
        if not os.path.exists(embeddings_images_output):
            os.makedirs(embeddings_images_output)

        embeddings_encoder_output = args.output + '/embeddings/encoder'
        if not os.path.exists(embeddings_encoder_output):
            os.makedirs(embeddings_encoder_output)

    if args.write_image_samples:
        images_output = args.output + '/images'
        if not os.path.exists(images_output):
            os.makedirs(images_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.load_checkpoint_path)
    LOG.info('Loading checkpoint...')
    if 'epoch' in checkpoint:
        LOG.info(' epoch %d', checkpoint['epoch'])
    if 'loss' in checkpoint:
        LOG.info(' loss %f', checkpoint['loss'])
    if 'acc' in checkpoint:
        LOG.info(' acc %f', checkpoint['acc'])
    if 'alphabet' in checkpoint:
        LOG.info(' alphabet %f', len(checkpoint['alphabet']))
    if 'state_dict' in checkpoint:
        LOG.info(' state_dict %f', len(checkpoint['state_dict']))
    if 'model_hyper_params' in checkpoint:
        LOG.info(' model_hyper_params %f',
                 checkpoint['model_hyper_params'])

    test_dataset, test_loader = get_datasets(args)

    model = AlignOcrModel(args, test_dataset.alphabet)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    model.eval()

    LOG.info('...Images to process %d', len(test_loader.dataset))

    display_hyp = True
    batch_size_total = 0
    correct_total = 0
    correct_total_view = 0

    with torch.no_grad():
        t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
        for sample_ctr, sample in enumerate(t):

            if sample_ctr % args.test_display_mod == 0:
                display_hyp = True

            sample = move_to_cuda(sample)

            targets = sample['target']
            target_lengths = sample['target_length']
            group_id = sample['group_id']

            batch_size = targets.size(0)
            batch_size_total += batch_size

            batch_shape = sample['batch_shape']
            LOG.debug('batch (img_cnt, sub_cnt, c, h, w)%s', batch_shape)

            net_meta = model(
                sample['net_input']['src_tokens'])

            logits = net_meta['logits']
            LOG.debug('logits (batch * image_cnt, vocab) %s', logits.shape)

            encoder_out = net_meta['embeddings']  # .squeeze()
            LOG.debug('embeddings (batch * image_cnt, embed_size) %s',
                      encoder_out.shape)

            encoder_out = encoder_out.squeeze()
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
                encoder_out = encoder_out.unsqueeze(0)

            logits_view = logits.view(
                batch_shape[0], batch_shape[1], logits.size(-1))
            LOG.debug('logits view (batch, image_cnt, vocab) %s',
                      logits_view.shape)

            encoder_out_view = encoder_out.view(
                batch_shape[0], batch_shape[1], encoder_out.size(-1))
            LOG.debug('embeddings view (batch, image_cnt, vocab) %s',
                      encoder_out_view.shape)

            image_list = sample['net_input']['src_tokens'].cpu().numpy()

            for logit_idx in range(batch_shape[0]):
                image_id = sample['image_id'][logit_idx]
                curr_logit = logits_view[logit_idx:logit_idx+1].squeeze()
                curr_encoder = encoder_out_view[logit_idx:logit_idx +
                                                1].squeeze()
                if len(curr_logit.shape) == 1:
                    curr_logit = curr_logit.unsqueeze(0)
                    curr_encoder = curr_encoder.unsqueeze(0)

                curr_target = targets[logit_idx *
                                      batch_shape[1]: logit_idx *
                                      batch_shape[1] + batch_shape[1]]

                _, curr_pred = curr_logit.topk(1, 1, True, True)
                curr_pred = curr_pred.t()

                utf8_target = []
                for curr_target_item in curr_target:  # .data.numpy():
                    utf8_target.append(model.alphabet[curr_target_item])
                utf8_ref_text = ''.join(utf8_target)

                if args.write_metadata:
                    decode_encoder_metadata = {}
                    decode_encoder_metadata['image_id'] = str(image_id)
                    decode_encoder_metadata['utf8_ref_text'] = utf8_ref_text
                    decode_encoder_metadata['target'] = curr_target.cpu(
                    ).numpy()
                    decode_encoder_metadata['encoder'] = curr_encoder.cpu(
                    ).numpy()
                    np.savez_compressed(os.path.join(embeddings_encoder_output, str(image_id) + '.npz'),
                                        allow_pickle=True,
                                        metadata=decode_encoder_metadata)

                    decode_images_metadata = {}
                    decode_images_metadata['image_id'] = str(image_id)
                    decode_images_metadata['utf8_ref_text'] = utf8_ref_text
                    decode_images_metadata['target'] = curr_target.cpu(
                    ).numpy()
                    decode_images_metadata['image'] = image_list[logit_idx]
                    np.savez_compressed(os.path.join(embeddings_images_output, str(image_id) + '.npz'),
                                        allow_pickle=True,
                                        metadata=decode_images_metadata)

                if display_hyp and logit_idx < 5:
                    LOG.info('batch %d, item %d', sample_ctr, logit_idx)
                    LOG.info('image_id %s', image_id)
                    LOG.info('utf8_ref_text %s', utf8_ref_text)
                    LOG.info('target %s', curr_target)
                    LOG.info('image %s', image_list[logit_idx].shape)
                    LOG.info('encoder %s', curr_encoder.shape)
                    LOG.info('logit %s', curr_logit.shape)

                    asci_pred = []
                    for curr_pred_item in curr_pred[0]:  # .data.numpy():
                        asci_pred.append(model.alphabet[curr_pred_item])
                    LOG.info('predict %s', ''.join(asci_pred))

                    asci_target = []
                    for curr_target_item in curr_target:  # .data.numpy():
                        asci_target.append(model.alphabet[curr_target_item])
                    LOG.info(' target %s', ''.join(asci_target))

                    if args.write_image_samples:
                        for logit_idx in range(image_list.shape[0]):
                            curr_target = targets[logit_idx *
                                                  batch_shape[1]: logit_idx *
                                                  batch_shape[1] + batch_shape[1]]
                            asci_target = []
                            for curr_target_item in curr_target:
                                asci_target.append(
                                    model.alphabet[curr_target_item])
                            for image_idx in range(image_list.shape[1]):
                                image = np.uint8(
                                    image_list[logit_idx][image_idx].transpose((1, 2, 0)) * 255)
                                # rgb to bgr
                                image = image[:, :, ::-1].copy()
                                curr_out_path = os.path.join(images_output, str(sample_ctr) + '_' +
                                                             str(logit_idx) + '_' + str(image_idx) +
                                                             '_' + asci_target[image_idx] + '.png')
                                cv2.imwrite(curr_out_path, image)

                correct_view = curr_pred.eq(
                    curr_target.view(1, -1).expand_as(curr_pred))
                correct_k_view = correct_view[:1].view(
                    -1).float().sum(0, keepdim=True)
                correct_total_view += int(correct_k_view)

            display_hyp = False

        LOG.info('Test images %d, correct %d, accuracy %.2f',
                 batch_size_total, correct_total_view, (correct_total_view/batch_size_total))

    LOG.info('...complete, time %s', time.process_time() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
