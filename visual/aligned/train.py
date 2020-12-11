""" Pre-train a visual embedding model."""
import os
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from vis_align_dataset import ImageSynthDataset, ImageGroupSampler, image_collater

import torch.nn.functional as F
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import sys
import argparse
import time
from PIL import Image
import cv2
import csv
import os
import unicodedata

from fairseq.data import Dictionary
from fairseq.modules import AlignOcrModel

import logging
LOG = logging.getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict', type=str,
                        help='Input dictionary')

    parser.add_argument('--train', type=str,
                        default='', help='Input text')
    parser.add_argument('--train-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument("--train-max-text-width", type=int,
                        default=60, help="max text width")
    parser.add_argument("--train-min-text-width", type=int,
                        default=1, help="max text width")

    parser.add_argument('--valid', type=str,
                        default='', help='Input text')
    parser.add_argument('--valid-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument("--valid-max-text-width", type=int,
                        default=60, help="max text width")
    parser.add_argument("--valid-min-text-width", type=int,
                        default=1, help="max text width")

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument('--augment', action='store_true',
                        help='train with augmentation')

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
    parser.add_argument("--epochs", type=int,
                        default=150, help="Nbr epochs")

    parser.add_argument("--lr", type=float,
                        default=1e-3, help="learning rate")
    parser.add_argument("--min-lr", type=float,
                        default=1e-7, help="Minimum learning rate")
    parser.add_argument("--patience", type=int,
                        default=5, help="patience before lr reduce")

    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="encoder dimension")
    parser.add_argument("--image-embed-dim", type=int,
                        default=512, help="image embedding dimension")

    parser.add_argument("--load-checkpoint-path", type=str,
                        default=None, help="Input checkpoint path")

    parser.add_argument('--write-image-samples', action='store_true',
                        help='write image samples')
    parser.add_argument("--write-image-mod", type=int,
                        default=3000, help="write images mod")

    parser.add_argument("--valid-display-mod", type=int,
                        default=3000, help="validation display mod")

    parser.add_argument('--image-verbose', action='store_true',
                        help='more debug info')

    args = parser.parse_args(argv)
    for arg in vars(args):
        LOG.info('%s %s', arg, getattr(args, arg))

    return args


class StrongAugment(object):

    def __init__(self):
        LOG.info('...using strong augment')
        def sometimes(aug): return iaa.Sometimes(.90, aug)

        seq = iaa.Sequential(
            [
                sometimes(iaa.CropAndPad(
                    percent=(-0.03, 0.03),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    rotate=(-2, 2),
                    shear=(-2, 2),
                )),
                # execute 0 to 2 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 2), [
                    iaa.OneOf([
                        # blur images with a sigma between 0 and 3.0
                        iaa.GaussianBlur((0, 3.0)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Affine(shear=(-1, 1)),
                    iaa.JpegCompression(compression=(70, 99)),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(
                        0.75, 1.5)),  # sharpen images
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.05*255), per_channel=0.5),
                    iaa.OneOf([
                        # randomly remove up to 10% of the pixels
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(
                            0.02, 0.05), per_channel=0.2),
                    ]),
                    # invert color channels
                    iaa.Invert(0.05, per_channel=True),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-10, 10), per_channel=0.5),
                    # change hue and saturation
                    iaa.AddToHueAndSaturation((-20, 20)),
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply(
                                (0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    # improve or worsen the contrast
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    # move pixels locally around (with random strengths)
                    sometimes(iaa.ElasticTransformation(
                        alpha=(0.5, 3.5), sigma=0.25)),
                ],
                    random_order=True
                )
            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug


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


def save_checkpoint(args, save_best, acc,
                    epoch, loss, model, optimizer, train_dataset, ckpts_output):

    if save_best:
        LOG.info('...checkpoint: saving new best %f', acc)
        torch.save({'epoch': epoch,
                    'loss': loss,
                    'acc': acc,
                    'vocab': train_dataset.vocab,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    },
                   os.path.join(ckpts_output, 'model_ckpt_best.pth'))

    torch.save({'epoch': epoch,
                'loss': loss,
                'acc': acc,
                'vocab': train_dataset.vocab,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                },
               os.path.join(ckpts_output, 'model_ckpt_last.pth'))


def get_datasets(args, cache_output):

    vocab = Dictionary.load(args.dict)

    LOG.info('loading dictionary %s', args.dict)
    for idx, char in enumerate(vocab.symbols):
        if idx < 10 or idx > len(vocab.symbols) - 10 - 1:
            LOG.info('...indicies %d, symbol %s, count %d', vocab.indices[vocab.symbols[idx]],
                     vocab.symbols[idx], vocab.count[idx])

    if args.augment:
        train_transform = transforms.Compose([
            StrongAugment(),
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    if args.write_image_samples:
        train_cache_output = cache_output
    else:
        train_cache_output = None

    train_dataset = ImageSynthDataset(
        text_file_path=args.train,
        font_file=args.train_font,
        transform=train_transform,
        vocab=vocab,
        max_text_width=args.train_max_text_width,
        min_text_width=args.train_min_text_width,
        image_height=args.image_height,
        image_width=args.image_width,
        image_cache=None,
        cache_output=train_cache_output)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=ImageGroupSampler(
                                                   train_dataset, rand=True),
                                               collate_fn=lambda b: image_collater(
                                                   b),
                                               num_workers=args.num_workers, pin_memory=True,
                                               drop_last=True)

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    valid_dataset = ImageSynthDataset(
        text_file_path=args.valid,
        font_file=args.valid_font,
        transform=valid_transform,
        vocab=vocab,
        max_text_width=args.valid_max_text_width,
        min_text_width=args.valid_min_text_width,
        image_height=args.image_height,
        image_width=args.image_width,
        image_cache=train_dataset.image_cache)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                               sampler=ImageGroupSampler(
                                                   valid_dataset, rand=False),
                                               collate_fn=lambda b: image_collater(
                                                   b),
                                               num_workers=args.num_workers, pin_memory=False,
                                               drop_last=False)

    return train_dataset, train_loader, valid_dataset, valid_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        # correct_res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args, model, train_loader, optimizer, device, classification_loss,
          sample, epoch, batch_iter, samples_train_output):
    model.train()

    iteration_start = time.time()

    sample = move_to_cuda(sample)
    batch_size = len(sample['net_input']['src_tokens'])

    targets = sample['target']
    target_lengths = sample['target_length']
    group_id = sample['group_id']

    batch_shape = sample['batch_shape']

    # Input: [32, 68, 3, 32, 32] batch, count, image
    if args.write_image_samples and batch_iter % args.write_image_mod == 0:
        image_list = sample['net_input']['src_tokens'].cpu().numpy()
        for logit_idx in range(image_list.shape[0]):
            curr_target = targets[logit_idx *
                                  batch_shape[1]: logit_idx *
                                  batch_shape[1] + batch_shape[1]]
            asci_target = []
            for curr_target_item in curr_target:
                asci_target.append(
                    model.vocab[curr_target_item])
            for image_idx in range(image_list.shape[1]):
                image = np.uint8(
                    image_list[logit_idx][image_idx].transpose((1, 2, 0)) * 255)
                image = image[:, :, ::-1].copy()  # rgb to bgr
                curr_out_path = os.path.join(samples_train_output, str(epoch) + '_' +
                                             str(batch_iter) + '_' + str(logit_idx) + '_' + str(image_idx) +
                                             '_' + asci_target[image_idx] + '.png')
                cv2.imwrite(curr_out_path, image)

    optimizer.zero_grad()

    net_meta = model(
        sample['net_input']['src_tokens'])  # .cuda())  # , sample['net_input']['src_widths'].cuda())

    logits = net_meta['logits']

    loss = classification_loss(
        logits, targets.to(device))

    acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

    loss.backward()
    optimizer.step()

    loss_val = float(loss)

    duration = time.time() - iteration_start
    examples_per_sec = batch_size / duration
    sec_per_batch = float(duration)

    max_target_len = max(target_lengths.cpu().numpy())

    if batch_iter % 10 == 0 and batch_iter > 0:
        data_len = len(train_loader)
        LOG.info("Epoch: %d (%d/%d), Input: %s, CNN %s, Emb %s, Logits %s, Acc_1 %.2f, Acc_5 %.2f, Loss: %.4f, LR: %.6f, ex/sec: %.1f, sec/batch: %.2f",
                 epoch, batch_iter + 1 % data_len, data_len,  # epoch
                 net_meta['input_shape'],
                 net_meta['encoder_cnn_shape'],
                 list(net_meta['embeddings'].shape),
                 # list(net_meta['encoder_out'].shape),
                 list(logits.shape),
                 acc1[0],
                 acc5[0],
                 loss_val,  # loss
                 get_lr(optimizer),
                 examples_per_sec, sec_per_batch)

    return loss_val


def validate(args, model, valid_loader, epoch, samples_valid_output):
    model.eval()
    valid_display = True
    LOG.info('...validation images to process %d', len(valid_loader.dataset))
    with torch.no_grad():

        batch_size_total = 0
        correct_total = 0
        sample_total = 0
        correct_total_view = 0

        t = tqdm(iter(valid_loader), leave=False, total=len(valid_loader))
        for sample_ctr, sample in enumerate(t):

            if sample_ctr % args.valid_display_mod == 0:
                valid_display = True

            sample = move_to_cuda(sample)

            targets = sample['target']
            target_lengths = sample['target_length']
            group_id = sample['group_id']

            batch_size = targets.size(0)  # of targets
            batch_size_total += batch_size  # len(sample)
            sample_total += len(sample['net_input']['src_tokens'])

            batch_shape = sample['batch_shape']

            net_meta = model(
                sample['net_input']['src_tokens'])

            logits = net_meta['logits']

            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            correct_total += int(correct_k)

            logits_view = logits.view(
                batch_shape[0], batch_shape[1], logits.size(-1))
            for logit_idx in range(batch_shape[0]):
                curr_logit = logits_view[logit_idx:logit_idx+1].squeeze()
                curr_target = targets[logit_idx *
                                      batch_shape[1]: logit_idx *
                                      batch_shape[1] + batch_shape[1]]

                _, curr_pred = curr_logit.topk(1, 1, True, True)
                curr_pred = curr_pred.t()

                if valid_display and logit_idx < 3:
                    LOG.info('batch %d, item %d', sample_ctr, logit_idx)

                    asci_pred = []
                    for curr_pred_item in curr_pred[0]:  # .data.numpy():
                        asci_pred.append(
                            model.vocab[curr_pred_item])
                    LOG.info('predict %s', ''.join(asci_pred))

                    asci_target = []
                    for curr_target_item in curr_target:  # .data.numpy():
                        asci_target.append(
                            model.vocab[curr_target_item])
                    LOG.info(' target %s', ''.join(asci_target))

                    if args.write_image_samples:
                        image_list = sample['net_input']['src_tokens'].cpu(
                        ).numpy()
                        for logit_idx in range(image_list.shape[0]):
                            curr_target = targets[logit_idx *
                                                  batch_shape[1]: logit_idx *
                                                  batch_shape[1] + batch_shape[1]]
                            asci_target = []
                            for curr_target_item in curr_target:
                                asci_target.append(
                                    model.vocab[curr_target_item])
                            for image_idx in range(image_list.shape[1]):
                                image = np.uint8(
                                    image_list[logit_idx][image_idx].transpose((1, 2, 0)) * 255)
                                # rgb to bgr
                                image = image[:, :, ::-1].copy()
                                curr_out_path = os.path.join(samples_valid_output, str(epoch) + '_' +
                                                             str(sample_ctr) + '_' + str(logit_idx) + '_' + str(image_idx) +
                                                             '_' + asci_target[image_idx] + '.png')
                                cv2.imwrite(curr_out_path, image)

                correct_view = curr_pred.eq(
                    curr_target.view(1, -1).expand_as(curr_pred))
                correct_k_view = correct_view[:1].view(
                    -1).float().sum(0, keepdim=True)
                correct_total_view += int(correct_k_view)

                valid_display = False

        LOG.info('Validation samples %d, images %d, correct %d, accuracy %.2f',
                 sample_total, batch_size_total, correct_total, (correct_total/batch_size_total))

        LOG.info('2 Validation images %d, correct %d, accuracy %.2f',
                 sample_total, correct_total_view, (correct_total_view/batch_size_total))

        acc = correct_total/batch_size_total
        return acc


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

    ckpts_output = args.output + '/checkpoints'
    if not os.path.exists(ckpts_output):
        os.makedirs(ckpts_output)

    cache_output = args.output + '/cache'
    if not os.path.exists(cache_output):
        os.makedirs(cache_output)

    samples_train_output = args.output + '/train'
    if not os.path.exists(samples_train_output):
        os.makedirs(samples_train_output)

    samples_valid_output = args.output + '/valid'
    if not os.path.exists(samples_valid_output):
        os.makedirs(samples_valid_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader, valid_dataset, valid_loader = get_datasets(
        args, cache_output)

    model = AlignOcrModel(args, train_dataset.vocab)
    model.to(device)

    classification_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=args.patience, min_lr=args.min_lr, verbose=True)

    if args.load_checkpoint_path:
        if os.path.isfile(args.load_checkpoint_path):
            checkpoint = torch.load(args.load_checkpoint_path)
            LOG.info('Loading checkpoint...')
            LOG.info(' epoch %d', checkpoint['epoch'])
            LOG.info(' loss %f', checkpoint['loss'])
            LOG.info(' acc %f', checkpoint['acc'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            LOG.info('...checkpoint not found %s', args.load_checkpoint_path)

    best_accuracy = None

    for epoch in range(args.epochs):

        model.train()

        for batch_iter, sample in enumerate(train_loader):

            loss_val = train(args, model, train_loader, optimizer, device, classification_loss,
                             sample, epoch, batch_iter, samples_train_output)

        acc = validate(args, model, valid_loader, epoch, samples_valid_output)
        scheduler.step(acc)

        save_best = False
        if best_accuracy is None:
            best_accuracy = acc
            save_best = True
        elif acc > best_accuracy:
            best_accuracy = acc
            save_best = True

        save_checkpoint(args, save_best, acc,
                        epoch, loss_val, model, optimizer, train_dataset, ckpts_output)

    LOG.info('...complete, time %s', time.process_time() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
