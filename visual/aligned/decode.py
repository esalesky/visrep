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
import csv

from fairseq.data import Dictionary
from fairseq.modules.vis_align_ocr import AlignOcrModel

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
    
# no longer exposed: built for batch-size=1
#    parser.add_argument("--batch-size", type=int,
#                        default=1, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="Nbr dataset workers")

    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="encoder dimension")
    parser.add_argument("--image-embed-dim", type=int,
                        default=512, help="image embedding dimension")

    parser.add_argument("--load-checkpoint-path", type=str,
                        default=None, help="Input checkpoint path")

    parser.add_argument("--test-display-mod", type=int,
                        default=10000, help="test display mod")

    parser.add_argument('--image-verbose', action='store_true',
                        help='more debug info')

    args = parser.parse_args(argv)
    for arg in vars(args):
        LOG.info('%s %s', arg, getattr(args, arg))

    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def np_norm(np_embed_norm):
    np_embed_norm_copy = np.copy(np_embed_norm)
    np_norm_val = np.linalg.norm(np_embed_norm_copy,
                                 keepdims=True)
    np_embed_norm_copy /= np_norm_val
    return np_embed_norm_copy


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

    vocab = Dictionary.load(args.dict)

    for idx, char in enumerate(vocab.symbols):
        if idx < 10 or idx > len(vocab.symbols) - 10 - 1:
            LOG.info('...indicies %d, symbol %s, count %d', vocab.indices[vocab.symbols[idx]],
                     vocab.symbols[idx], vocab.count[idx])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = ImageSynthDataset(
        text_file_path=args.test,
        font_file=args.test_font,
        transform=test_transform,
        vocab=vocab,
        max_text_width=args.test_max_text_width,
        min_text_width=args.test_min_text_width,
        image_height=args.image_height,
        image_width=args.image_width,
        image_cache=None)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
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


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.load_checkpoint_path)
    LOG.info('Loading checkpoint...')
    if 'epoch' in checkpoint:
        LOG.info(' epoch %d', checkpoint['epoch'])
    if 'loss' in checkpoint:
        LOG.info(' loss %f', checkpoint['loss'])
    if 'acc' in checkpoint:
        LOG.info(' acc %f', checkpoint['acc'])
    if 'vocab' in checkpoint:
        LOG.info(' vocab %f', len(checkpoint['vocab']))
    if 'state_dict' in checkpoint:
        LOG.info(' state_dict %f', len(checkpoint['state_dict']))
    if 'model_hyper_params' in checkpoint:
        LOG.info(' model_hyper_params %f',
                 checkpoint['model_hyper_params'])

    test_dataset, test_loader = get_datasets(args)

    model = AlignOcrModel(args, test_dataset.vocab)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    model.eval()

    LOG.info('...Images to process %d', len(test_loader.dataset))

    display_hyp = True
    batch_size_total = 0
    correct_total = 0
    correct_total_view = 0

    all_embeds = []
    all_labels = []

    embed_file = open(args.output + "/embeddings.txt", "w")
    norm_file  = open(args.output + "/norm_embeddings.txt", "w")
    embed_file_writer = csv.writer(embed_file, delimiter=' ')
    norm_file_writer  = csv.writer(norm_file, delimiter=' ')
    text_row = [len(test_dataset.vocab.symbols), args.image_embed_dim]
    embed_file_writer.writerow(text_row)
    norm_file_writer.writerow(text_row)
    
    
    # each sample should be a entry in the vocab, plus </s> because.
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

            net_meta = model(sample['net_input']['src_tokens'])

            # label
            label = sample['seed_text'][0].strip('</s>')
            
            # prediction
            logits = net_meta['logits']
            LOG.debug('logits (batch * image_cnt, vocab) %s', logits.shape)

            # the meat
            encoder_out = net_meta['embeddings']  # .squeeze()
            LOG.debug('embeddings (batch * image_cnt, embed_size) %s',
                      encoder_out.shape)
            encoder_out = encoder_out.squeeze()

            # the money
            encoder_out = encoder_out.squeeze()
            embed = encoder_out[0].cpu().numpy()
            norm_embed = np_norm(embed)

            # write to embed txt files
            text_row = [label]
            text_row = text_row + embed.tolist()
            embed_file_writer.writerow(text_row)

            text_row = [label]
            text_row = text_row + norm_embed.tolist()
            norm_file_writer.writerow(text_row)
            
            # append
            all_embeds.append(embed)
            all_labels.append(label)

    embed_file.close()
    norm_file.close()


    # write to npz file
    all_embeds = np.array(all_embeds)
    all_labels = np.array(all_labels)
    print('feature shape {}, labels shape {}'.format(
        all_embeds.shape, all_labels.shape))
    np.savez_compressed(args.output + "/embeddings.npz",
                        features=all_embeds, labels=all_labels)
    

    LOG.info('...complete, time %s', time.process_time() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
