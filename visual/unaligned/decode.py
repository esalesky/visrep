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
from vis_unalign_dataset import ImageSynthDataset, ImageGroupSampler, image_collater
from tqdm import tqdm
import torch.nn.functional as F
import logging
import unicodedata

from fairseq.data import Dictionary
from vis_unalign_ocr import UnAlignOcrModel
from text_utils import compute_cer_wer, uxxxx_to_utf8, utf8_to_uxxxx

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

    parser.add_argument("--load-checkpoint-path", type=str,
                        default=None, help="Input checkpoint path")

    parser.add_argument('--write-image-samples', action='store_true',
                        help='write image samples')
    parser.add_argument("--write-test-image-mod", type=int,
                        default=500, help="write images mod")

    parser.add_argument('--write-metadata', action='store_true',
                        help='write metadata')

    parser.add_argument("--ocr-fract-width-perc", type=float,
                        default=0.30, help="OCR fractional max pool percentage for width")

    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="encoder dimension")

    parser.add_argument("--decoder-lstm-layers", type=int,
                        default=3, help="Number of LSTM layers in model")
    parser.add_argument("--decoder-lstm-units", type=int,
                        default=256, help="Number of LSTM hidden units ")  # 640 256
    parser.add_argument("--decoder-lstm-dropout", type=float,
                        default=0.50, help="Number of LSTM layers in model")

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
        image_width=None,
        image_cache=None,
        cache_output=None)

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

    model = UnAlignOcrModel(args, test_dataset.alphabet)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    model.eval()

    LOG.info('...Images to process %d', len(test_loader.dataset))

    total_ref_chars = 0
    total_ref_char_dist = 0
    total_ref_words = 0
    total_ref_word_dist = 0
    batch_size_total = 0

    with torch.no_grad():
        t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
        for sample_ctr, sample in enumerate(t):

            sample = move_to_cuda(sample)

            targets = sample['target']
            target_lengths = sample['target_length']
            group_id = sample['group_id']

            batch_size = targets.size(0)
            batch_size_total += batch_size

            batch_shape = sample['batch_shape']
            LOG.debug('batch (batch, c, h, w)%s', batch_shape)

            net_meta = model(
                sample['net_input']['src_tokens'])

            logits = net_meta['logits']
            LOG.debug('logits (step, batch, vocab) %s', logits.shape)

            encoder_out = net_meta['embeddings']  # .squeeze()
            LOG.debug('embeddings (step * batch, embed_size) %s',
                      encoder_out.shape)

            # calculate cer
            input_lengths = torch.full(
                (batch_size,), logits.size(0), dtype=torch.int32)
            hyp_transcriptions = model.decode(
                logits, input_lengths)
            for hyp_ctr in range(len(hyp_transcriptions)):
                hyp_text = hyp_transcriptions[hyp_ctr]
                uxxxx_hyp_text = utf8_to_uxxxx(hyp_text)

                ref_text = sample['seed_text'][hyp_ctr]
                uxxxx_ref_text = utf8_to_uxxxx(ref_text)

                ref_char_dist, ref_chars, ref_word_dist, ref_words = compute_cer_wer(
                    uxxxx_hyp_text, uxxxx_ref_text)

                total_ref_char_dist += ref_char_dist
                total_ref_chars += ref_chars
                total_ref_word_dist += ref_word_dist
                total_ref_words += ref_words

                if sample_ctr % args.write_test_image_mod == 0 and hyp_ctr < 5:
                    LOG.info('%s', sample['image_id'][hyp_ctr])
                    LOG.info('hyp: %s', hyp_text)
                    LOG.info('ref: %s', ref_text)

            image_list = sample['net_input']['src_tokens'].cpu().numpy()

            if args.write_image_samples and sample_ctr % args.write_test_image_mod == 0:
                image_list = sample['net_input']['src_tokens'].cpu().numpy()
                for logit_idx in range(image_list.shape[0]):

                    # -4 to remove </s>
                    target_text = sample['seed_text'][logit_idx][:-4]

                    image = np.uint8(
                        image_list[logit_idx].transpose((1, 2, 0)) * 255)
                    image = image[:, :, ::-1].copy()  # rgb to bgr
                    curr_out_path = os.path.join(images_output,
                                                 str(sample_ctr) + '_' + str(logit_idx) +
                                                 '_' + target_text + '.png')
                    LOG.info('write test sample %s to %s',
                             target_text, curr_out_path)
                    cv2.imwrite(curr_out_path, image)

                    if logit_idx > 5:
                        break

            logits_view = logits.permute(1, 0, 2)
            LOG.debug('logits view (batch, step, vocab) %s',
                      logits_view.shape)

            encoder_out_view = encoder_out.view(
                logits.shape[0], logits.shape[1], encoder_out.size(-1))
            LOG.debug('embeddings view (step, batch, embed_size) %s',
                      encoder_out_view.shape)
            encoder_out_view = encoder_out_view.permute(1, 0, 2)
            LOG.debug('embeddings permute (batch, step, embed_size) %s',
                      encoder_out_view.shape)

            target_start = 0
            for logit_idx in range(batch_shape[0]):
                image_id = sample['image_id'][logit_idx]

                curr_logit = logits_view[logit_idx:logit_idx+1].squeeze()
                curr_encoder = encoder_out_view[logit_idx:logit_idx +
                                                1].squeeze()
                # if len(curr_logit.shape) == 1:
                #    curr_encoder = curr_encoder.unsqueeze(0)

                target_width = target_lengths[logit_idx]
                curr_target = targets[target_start: target_start + target_width]

                target_start += target_width

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

                    LOG.debug('metadata: image_id %s', str(image_id))
                    LOG.debug('metadata: utf8_ref_text %s', utf8_ref_text)
                    LOG.debug('metadata: curr_target %s', curr_target)
                    LOG.debug('metadata: curr_encoder %s', curr_encoder.shape)
                    LOG.debug('metadata: image %s',
                              image_list[logit_idx].shape)

    LOG.info('Test images %d, cer %.2f, wer %.2f',
             batch_size_total, total_ref_char_dist/total_ref_chars, total_ref_word_dist/total_ref_words)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
