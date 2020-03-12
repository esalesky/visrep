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
from augment import OcrAug, ImageAug
from ocr_dataset import LineSynthDataset, LmdbDataset, ImageGroupSampler, image_collater

import torch.nn.functional as F
import logging
from text_utils import compute_cer_wer, uxxxx_to_utf8, utf8_to_uxxxx

from model import OcrDecoder, OcrEncoder, OcrModel
#from cnnlstm import CnnOcrModel

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str,
                        default='', help='Input train seed text or lmdb')
    parser.add_argument('--train-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument('--train-background', type=str,
                        default='', help='Input train background file')
    parser.add_argument("--train-max-image-width", type=int,
                        default=1500, help="max train image width")
    parser.add_argument("--train-min-image-width", type=int,
                        default=32, help="min train image width")
    parser.add_argument('--train-lmdb', action='store_true',
                        help='Train using lmdb')
    parser.add_argument("--train-split", type=str,
                        default='train', help="train lmdb split")
    parser.add_argument('--train-split-text', action='store_true',
                        help='Split synthetic text')
    parser.add_argument("--train-max-seed", type=int,
                        default=500000, help="max seed")
    parser.add_argument("--train-max-text-width", type=int,
                        default=60, help="max text width")
    parser.add_argument('--train-use-default-image', action='store_true',
                        help='Use default image ')

    parser.add_argument('--valid', type=str,
                        default='', help='Input train seed text or lmdb')
    parser.add_argument('--valid-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument('--valid-background', type=str,
                        default='', help='Input train background file')
    parser.add_argument("--valid-max-image-width", type=int,
                        default=1500, help="max train image width")
    parser.add_argument("--valid-min-image-width", type=int,
                        default=32, help="min train image width")
    parser.add_argument('--valid-lmdb', action='store_true',
                        help='Train using lmdb')
    parser.add_argument("--valid-split", type=str,
                        default='validation', help="train lmdb split")
    parser.add_argument('--valid-split-text', action='store_true',
                        help='Split synthetic text')
    parser.add_argument("--valid-max-seed", type=int,
                        default=500000, help="max seed")
    parser.add_argument("--valid-max-text-width", type=int,
                        default=60, help="max text width")
    parser.add_argument('--valid-use-default-image', action='store_true',
                        help='Use default image ')

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument('--augment', action='store_true',
                        help='train with augmentation')
    parser.add_argument('--use-font-chars', action='store_true',
                        help='Use chars from font')

    parser.add_argument("--image-height", type=int,
                        default=32, help="Image height")

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

    parser.add_argument('--save-images', action='store_true',
                        help='Save images to disk')
    parser.add_argument('--use-image-cache', action='store_true',
                        help='Cache images')
    parser.add_argument("--max-image-cache", type=int,
                        default=250000, help="max image cache")
    parser.add_argument("--max-cache-write", type=int,
                        default=5000, help="max cache write")

    parser.add_argument("--valid-batch-mod", type=int,
                        default=None, help="validation batch mod")
    parser.add_argument("--valid-epoch-mod", type=int,
                        default=None, help="validation epoch mod")

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


def train(epoch_iter, batch_iter, model, optimizer, sample, data_len, samples_train_output, curr_alphabet):
    iteration_start = time.time()

    # sample = move_to_cuda(sample)
    # input_tensor, target_transcription, input_tensor_widths, target_transcription_widths, input_tensor_shape = sample
    # batch_size = len(input_tensor_widths)

    sample = move_to_cuda(sample)
    batch_size = len(sample['net_input']['src_widths'])

    optimizer.zero_grad()

    # input_shape, net_output, model_output_actual_lengths, encoder_bridge, lstm_out, encoder_cnn = model(
    # print('... train cuda sample')
    # net_output, net_meta = model(
    # net_output, lstm_output_lengths = model(
    #    input_tensor.cuda(), input_tensor_widths.cuda())

    # net_output, lstm_output_lengths, net_meta = model(
    #    **sample['net_input'])
    net_meta = model(
        sample['net_input']['src_tokens'].cuda(), sample['net_input']['src_widths'].cuda())

    log_probs = F.log_softmax(net_meta['prob_output_view'], dim=2)
    targets = sample['target']
    target_lengths = sample['target_length']

    # targets = target_transcription
    # target_lengths = target_transcription_widths

    # CUDA, PyTorch native implementation: OK
    torch.backends.cudnn.enabled = False
    loss = F.ctc_loss(log_probs.to('cuda'),
                      targets.to('cuda'),
                      # lstm_output_lengths,
                      net_meta['lstm_output_lengths'],
                      target_lengths,
                      reduction='mean',
                      zero_infinity=True)
    # loss = F.ctc_loss(log_probs.to('cuda'),
    #                  targets,
    #                  # net_meta['lstm_output_lengths'],
    #                  lstm_output_lengths,
    #                  target_lengths,
    #                  reduction='mean',
    #                  zero_infinity=True)
    loss.backward()
    optimizer.step()

    loss_val = float(loss)

    duration = time.time() - iteration_start
    examples_per_sec = batch_size / duration
    sec_per_batch = float(duration)

    max_target_len = max(target_lengths.cpu().numpy())

    if batch_iter % 10 == 0 and batch_iter > 0:
        # LOG.info("Epoch: %d (%d/%d), Group: %s, Shape: %s, Target: %s, CNN %s, Bridge: %s, LSTM %s, Loss: %.4f, LR: %.6f, ex/sec: %.1f, sec/batch: %.2f",
        #          epoch_iter, batch_iter + 1 % data_len,  # epoch
        #          data_len,
        #          sample['group_id'],  # group
        #          sample['batch_shape'],  # shape
        #          max(target_lengths.cpu().numpy()),  # target
        #          # encoder_cnn.detach().cpu().numpy().shape,
        #          net_meta['cnn_shape'],  # cnn
        #          # encoder_bridge.detach().cpu().numpy().shape,
        #          net_meta['bridge_shape'],  # bridge
        #          # lstm_out.detach().cpu().numpy().shape,
        #          net_meta['lstm_output_shape'],  # lstm
        #          loss.item(),  # loss
        #          get_lr(optimizer), examples_per_sec, sec_per_batch)  # lr, ex_sec, sec_batch

        LOG.info("Epoch: %d (%d/%d), Target: %s, Input: %s, CNN %s, LSTM %s, Loss: %.4f, LR: %.6f, ex/sec: %.1f, sec/batch: %.2f",
                 epoch_iter, batch_iter + 1 % data_len,  data_len,  # epoch
                 max_target_len,  # target
                 net_meta['input_shape'],
                 net_meta['cnn_shape'],
                 net_meta['lstm_output_shape'],
                 loss_val,  # loss
                 get_lr(optimizer),
                 examples_per_sec, sec_per_batch)

    if batch_iter % 100 == 0:
        image_list = sample['net_input']['src_tokens'].cpu().numpy()
        label_list = sample['seed_text']
        # image_list = input_tensor.cpu().numpy()
        # target_np = target_transcription.data.cpu().numpy()

        # cur_target_offset = 0
        for img_idx, img in enumerate(image_list):
            if img_idx < 2:

                seed_text = label_list[img_idx]

                # ref_id = target_np[cur_target_offset:(
                #    cur_target_offset + target_transcription_widths.data[img_idx])]
                # uxxxx_ref_text = form_target_transcription(
                #    ref_id, curr_alphabet)
                # cur_target_offset += target_transcription_widths.data[img_idx]
                # seed_text = uxxxx_to_utf8(uxxxx_ref_text)

                image = np.uint8(img.transpose((1, 2, 0)) * 255)
                image = image[:, :, ::-1].copy()  # rgb to bgr
                (h, w) = image.shape[:2]

                outpath = samples_train_output + '/' + \
                    seed_text + '.png'
                cv2.imwrite(outpath, image)

    return loss_val


def form_target_transcription(target, alphabet):
    trans_list = []
    for i in target:
        if i in alphabet.idx_to_char:
            trans_list.append(alphabet.idx_to_char[i])
        else:
            # print('...validation char %s missing from train model ' % (i))
            trans_list.append(alphabet.idx_to_char[1])
    return ' '.join(trans_list)


def validate(epoch_iter, model, validloader, samples_valid_output):
    cer_running_avg = 0
    wer_running_avg = 0
    n_samples = 0
    loss_running_avg = 0
    total_val_images = 0

    model.eval()
    with torch.no_grad():
        print('...validate on %d images' % (len(validloader.dataset)))

        for sample_ctr, sample in enumerate(validloader):

            # sample = move_to_cuda(sample)
            # input_tensor, target_transcription, input_tensor_widths, target_transcription_widths, input_tensor_shape = sample
            # input_tensor.to('cuda')
            # input_tensor_widths.to('cuda')

            sample = move_to_cuda(sample)
            batch_size = len(sample['net_input']['src_widths'])
            # batch_size = len(input_tensor_widths)

            total_val_images += batch_size

            # input_shape, net_output, model_output_actual_lengths, encoder_shape, lstm_shape, encoder_cnn = model(
            # net_output, net_meta = model(
            # net_output, lstm_output_lengths = model(
            #    input_tensor.cuda(), input_tensor_widths.cuda())

            # net_output, lstm_output_lengths, net_meta = model(
            #    **sample['net_input'])
            # net_output, lstm_output_lengths = model(
            net_meta = model(
                sample['net_input']['src_tokens'].cuda(), sample['net_input']['src_widths'].cuda())

            log_probs = F.log_softmax(net_meta['prob_output_view'], dim=2)

            targets = sample['target']
            target_lengths = sample['target_length']

            # targets = target_transcription
            # target_lengths = target_transcription_widths

            # CUDA, PyTorch native implementation: OK
            torch.backends.cudnn.enabled = False
            loss = F.ctc_loss(log_probs.to('cuda'),
                              targets.to('cuda'),
                              net_meta['lstm_output_lengths'],
                              # lstm_output_lengths,
                              target_lengths,
                              reduction='mean',
                              zero_infinity=True)
            # loss = F.ctc_loss(log_probs.to('cuda'),
            #                  targets,
            #                  # net_meta['lstm_output_lengths'],
            #                  lstm_output_lengths,
            #                  target_lengths,
            #                  reduction='mean',
            #                  zero_infinity=True)

            n_samples += 1

            loss_val = float(loss)
            loss_running_avg += (loss_val -
                                 loss_running_avg) / n_samples

            # hyp_transcriptions = model.decode(
            #    net_output, lstm_output_lengths)  # net_meta['lstm_output_lengths'])  # model_output_actual_lengths)

            hyp_transcriptions = model.decode(
                net_meta['prob_output_view'], net_meta['lstm_output_lengths'])  # net_meta['lstm_output_lengths'])

            batch_cer = 0
            batch_wer = 0

            # target_np = target_transcription.data.cpu().numpy()
            # cur_target_offset = 0
            for hyp_ctr in range(len(hyp_transcriptions)):
                utf8_ref_text = sample['seed_text'][hyp_ctr]
                uxxxx_ref_text = utf8_to_uxxxx(utf8_ref_text)

                # ref_id = target_np[cur_target_offset:(
                #    cur_target_offset + target_transcription_widths.data[hyp_ctr])]
                # uxxxx_ref_text = form_target_transcription(
                #    ref_id, validloader.dataset.alphabet)  # visual
                # cur_target_offset += target_transcription_widths.data[hyp_ctr]

                # utf8_hyp_text = uxxxx_to_utf8(hyp_transcriptions[hyp_ctr])
                uxxxx_hyp_text = hyp_transcriptions[hyp_ctr]

                cer, wer = compute_cer_wer(
                    uxxxx_hyp_text, uxxxx_ref_text)

                batch_cer += cer
                batch_wer += wer

                if sample_ctr % 50 == 0:
                    # image_list = input_tensor.cpu().numpy()
                    # img = image_list[hyp_ctr]

                    image_list = sample['net_input']['src_tokens'].cpu(
                    ).numpy()

                    img = image_list[hyp_ctr]

                    if hyp_ctr < 2:
                        print('\nhyp', uxxxx_hyp_text)
                        print('ref', uxxxx_ref_text)
                        print('cer %f \twer %f' % (cer, wer))

                        seed_text = uxxxx_to_utf8(uxxxx_ref_text)

                        image = np.uint8(img.transpose((1, 2, 0)) * 255)
                        image = image[:, :, ::-1].copy()  # rgb to bgr
                        (h, w) = image.shape[:2]

                        outpath = samples_valid_output + '/' + \
                            seed_text + '.png'
                        cv2.imwrite(outpath, image)

            cer_running_avg += (batch_cer / batch_size -
                                cer_running_avg) / n_samples
            wer_running_avg += (batch_wer / batch_size -
                                wer_running_avg) / n_samples

    print("Validation Images: %d Loss: %f \tVal CER: %f\tVal WER: %f \tSamples: %d" %
          (total_val_images, loss_running_avg, cer_running_avg, wer_running_avg, n_samples))

    return cer_running_avg, wer_running_avg


def save_checkpoint(save_best, cer_running_avg, wer_running_avg,
                    epoch, loss, model, optimizer, train_dataset, ckpts_output):

    if save_best:
        print('...checkpoint: saving new best', cer_running_avg)
        torch.save({'epoch': epoch,
                    'loss': loss,
                    'cer': cer_running_avg,
                    'wer': wer_running_avg,
                    'alphabet': train_dataset.alphabet,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    },
                   os.path.join(ckpts_output, 'model_ckpt_best.pth'))

    torch.save({'epoch': epoch,
                'loss': loss,
                'cer': cer_running_avg,
                'wer': wer_running_avg,
                'alphabet': train_dataset.alphabet,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                },
               os.path.join(ckpts_output, 'model_ckpt_last.pth'))


def get_datasets(args):
    if args.augment:
        train_transform = transforms.Compose([
            # OcrAug(),
            ImageAug(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    if args.train_lmdb:
        train_dataset = LmdbDataset(args.train,
                                    args.train_split,
                                    train_transform,
                                    image_height=args.image_height,
                                    image_verbose=args.image_verbose,
                                    max_image_width=args.train_max_image_width)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   sampler=ImageGroupSampler(
                                                       train_dataset, rand=True),
                                                   collate_fn=lambda b: image_collater(
                                                       b, args.image_verbose),
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   drop_last=True)
    else:
        train_dataset = LineSynthDataset(text_file_path=args.train,
                                         font_file_path=args.train_font,
                                         bkg_file_path=args.train_background,
                                         image_height=args.image_height,
                                         transform=train_transform,
                                         max_seed=args.train_max_seed,
                                         use_default_image=args.train_use_default_image,
                                         max_image_width=args.train_max_image_width,
                                         min_image_width=args.train_min_image_width,
                                         split_text=args.train_split_text,
                                         max_text_width=args.train_max_text_width,
                                         augment=args.augment,
                                         use_font_chars=args.use_font_chars,
                                         image_verbose=args.image_verbose)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   sampler=ImageGroupSampler(
                                                       train_dataset, rand=True),
                                                   collate_fn=lambda b: image_collater(
                                                       b, args.image_verbose),
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   drop_last=True)
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    if args.valid_lmdb:
        valid_dataset = LmdbDataset(args.valid,
                                    args.valid_split,
                                    valid_transform,
                                    image_height=args.image_height,
                                    image_verbose=args.image_verbose,
                                    alphabet=train_dataset.alphabet,
                                    max_image_width=args.valid_max_image_width)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8,
                                                   sampler=ImageGroupSampler(
                                                       valid_dataset, rand=False),
                                                   collate_fn=lambda b: image_collater(
                                                       b, args.image_verbose),
                                                   num_workers=0, pin_memory=False,
                                                   drop_last=False)
    else:
        valid_dataset = LineSynthDataset(text_file_path=args.valid,
                                         font_file_path=args.valid_font,
                                         bkg_file_path=args.valid_background,
                                         image_height=args.image_height,
                                         transform=valid_transform,
                                         max_seed=args.valid_max_seed,
                                         use_default_image=args.valid_use_default_image,
                                         max_image_width=args.valid_max_image_width,
                                         min_image_width=args.valid_min_image_width,
                                         split_text=args.valid_split_text,
                                         max_text_width=args.valid_max_text_width,
                                         use_font_chars=args.use_font_chars,
                                         image_verbose=args.image_verbose)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                   sampler=ImageGroupSampler(
                                                       valid_dataset, rand=False),
                                                   collate_fn=lambda b: image_collater(
                                                       b, args.image_verbose),
                                                   num_workers=args.num_workers, pin_memory=False,
                                                   drop_last=False)

    return train_dataset, train_loader, valid_dataset, valid_loader


def main(args):
    start_time = time.clock()

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

    samples_train_output = args.output + '/train'
    if not os.path.exists(samples_train_output):
        os.makedirs(samples_train_output)

    samples_valid_output = args.output + '/valid'
    if not os.path.exists(samples_valid_output):
        os.makedirs(samples_valid_output)

    train_cache_output = args.output + '/cache'
    if not os.path.exists(train_cache_output):
        os.makedirs(train_cache_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader, valid_dataset, valid_loader = get_datasets(
        args)

    # if args.image_verbose:
    tgt_dict = train_dataset.alphabet
    print('...Dictionary train', len(tgt_dict))
    for i in range(len(tgt_dict)):
        if i < 5:
            print(i, tgt_dict.idx_to_char[i])
        elif i > len(tgt_dict) - 5:
            print(i, tgt_dict.idx_to_char[i])

    valid_dict = valid_dataset.alphabet
    print('...Dictionary valid', len(valid_dict))
    for i in range(len(valid_dict)):
        if i < 5:
            print(i, valid_dict.idx_to_char[i])
        elif i > len(valid_dict) - 5:
            print(i, valid_dict.idx_to_char[i])

    encoder = OcrEncoder(args)
    decoder = OcrDecoder(args, train_dataset.alphabet)
    model = OcrModel(args, encoder, decoder, train_dataset.alphabet)

    # model = CnnOcrModel(
    #     num_in_channels=3,
    #     input_line_height=args.image_height,
    #     rds_line_height=args.image_height,
    #     lstm_input_dim=args.encoder_dim,
    #     num_lstm_layers=args.decoder_lstm_layers,
    #     num_lstm_hidden_units=args.decoder_lstm_units,
    #     p_lstm_dropout=0.5,
    #     alphabet=train_dataset.alphabet,
    #     multigpu=False)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, min_lr=args.min_lr, verbose=True)

    if args.load_checkpoint_path:
        if os.path.isfile(args.load_checkpoint_path):
            checkpoint = torch.load(args.load_checkpoint_path)
            LOG.info('Loading checkpoint...')
            LOG.info(' epoch %d', checkpoint['epoch'])
            LOG.info(' loss %f', checkpoint['loss'])
            LOG.info(' cer %f', checkpoint['cer'])
            LOG.info(' wer %f', checkpoint['wer'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            checkpoint_alphabet = checkpoint['alphabet']

        else:
            print('...checkpoint not found', args.load_checkpoint_path)

    #summary(model, [(3, args.image_height, 800), ([8, 8])])

    cer_running_avg = 9999
    wer_running_avg = 9999
    best_cer = None
    model.train()
    for epoch in range(args.epochs):
        for batch_iter, sample in enumerate(train_loader):

            loss = train(epoch, batch_iter, model, optimizer, sample, len(
                train_loader), samples_train_output, train_dataset.alphabet)

            if args.valid_batch_mod:
                if (batch_iter % args.valid_batch_mod) == 0 and batch_iter > 0:
                    cer_running_avg, wer_running_avg = validate(
                        epoch, model, valid_loader, samples_valid_output)

                    model.train()

                    save_best = False
                    if best_cer is None:
                        best_cer = cer_running_avg
                        save_best = True
                    elif cer_running_avg < best_cer:
                        best_cer = cer_running_avg
                        save_best = True

                    save_checkpoint(save_best, cer_running_avg, wer_running_avg,
                                    epoch, loss, model, optimizer, train_dataset, ckpts_output)

        if args.valid_epoch_mod:
            if ((epoch + 1) % args.valid_epoch_mod) == 0:
                cer_running_avg, wer_running_avg = validate(
                    epoch, model, valid_loader, samples_valid_output)

                model.train()

                scheduler.step(cer_running_avg)

                save_best = False
                if best_cer is None:
                    best_cer = cer_running_avg
                    save_best = True
                elif cer_running_avg < best_cer:
                    best_cer = cer_running_avg
                    save_best = True

                save_checkpoint(save_best, cer_running_avg, wer_running_avg,
                                epoch, loss, model, optimizer, train_dataset, ckpts_output)

    LOG.info('...complete, time %s', time.clock() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
