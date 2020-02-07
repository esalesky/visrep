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
from augment import ImageAug
from dataset import ImageDataset, GroupedSampler, SortByWidthCollater
# from models import VisualNet, Softmax, VisualTrainer
import torch.nn.functional as F
import logging
from text_utils import compute_cer_wer, uxxxx_to_utf8, utf8_to_uxxxx

from torch.nn.utils.rnn import pack_padded_sequence as rnn_pack
from torch.nn.utils.rnn import pad_packed_sequence as rnn_unpack

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str,
                        default='', help='Input seed text')
    parser.add_argument('--valid', type=str,
                        default='', help='Input seed text')
    parser.add_argument('--train-font', type=str,
                        default='', help='Input train font file')
    parser.add_argument('--valid-font', type=str,
                        default='', help='Input validation font file')
    parser.add_argument('--output', type=str,
                        default='', help='Output directory')
    parser.add_argument('--augment', action='store_true',
                        help='train with augmentation')
    parser.add_argument("--image-height", type=int,
                        default=30, help="Image height")
    parser.add_argument("--batch-size", type=int,
                        default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int,
                        default=150, help="Nbr epochs")
    parser.add_argument("--max-image-cache", type=int,
                        default=250000, help="max image cache")
    parser.add_argument("--max-cache-write", type=int,
                        default=5000, help="max cache write")
    parser.add_argument("--max-seed", type=int,
                        default=500000, help="max seed")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="Nbr dataset workers")
    parser.add_argument("--lr", type=float,
                        default=1e-3, help="learning rate")
    parser.add_argument("--min-lr", type=float,
                        default=1e-7, help="Minimum learning rate")
    parser.add_argument("--encoder-dim", type=int,
                        default=512, help="Encoder dim")
    parser.add_argument("--decoder-lstm-layers", type=int,
                        default=3, help="Number of LSTM layers in model")
    parser.add_argument("--decoder-lstm-units", type=int,
                        default=640, help="Number of LSTM hidden units ")  # 640
    parser.add_argument("--decoder-lstm-dropout", type=float,
                        default=0.50, help="Number of LSTM layers in model")
    parser.add_argument('--image-verbose', action='store_true',
                        help='Debug info')
    parser.add_argument('--save-images', action='store_true',
                        help='Save images to disk')
    parser.add_argument('--use-image-cache', action='store_true',
                        help='Cache images')

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


class OCRCRNNModel(torch.nn.Module):
    def __init__(self, encoder, decoder, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, src_widths):
        encoder_out = self.encoder(src_tokens, src_widths)
        decoder_out = self.decoder(encoder_out)

        return decoder_out

    def decode(self, model_output, batch_actual_timesteps):
        # start_decode = datetime.now()
        min_prob_thresh = 3 * 1 / len(self.dictionary)

        T = model_output.size()[0]
        B = model_output.size()[1]

        prev_char = ['' for _ in range(B)]
        result = ['' for _ in range(B)]

        for t in range(T):

            gpu_argmax = False
            model_output_at_t_cpu = model_output.data[t].cpu().numpy()
            argmaxs = model_output_at_t_cpu.max(1).flatten()
            argmax_idxs = model_output_at_t_cpu.argmax(1).flatten()

            for b in range(B):
                # Only look at valid model output for this batch entry
                if t >= batch_actual_timesteps[b]:
                    continue

                if argmax_idxs[b] == 0:  # CTC Blank
                    prev_char[b] = ''
                    continue

                # Heuristic
                # If model is predicting very low probability for all letters in alphabet, treat that the
                # samed as a CTC blank
                if argmaxs[b] < min_prob_thresh:
                    prev_char[b] = ''
                    continue

                char = self.dictionary.idx_to_char[argmax_idxs[b]]

                if prev_char[b] == char:
                    continue

                result[b] += char
                prev_char[b] = char

                # Add a space to all but last iteration
                # only need if chars encoded as uxxxx
                if t != T - 1:
                    result[b] += ' '

        # Strip off final token-stream space if needed
        for b in range(B):
            if len(result[b]) > 0 and result[b][-1] == ' ':
                result[b] = result[b][:-1]

        return result  # , uxxx_result


class OCRCNNEncoder(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(3, 64),
            *self.ConvBNReLU(64, 64),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(64, 128),
            *self.ConvBNReLU(128, 128),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(128, 256),
            *self.ConvBNReLU(256, 256),
            *self.ConvBNReLU(256, 256)
        )

        # We need to calculate cnn output size to construct the bridge layer
        fake_input_width = 800
        print('Fake input width %d' % (fake_input_width))
        cnn_out_h, cnn_out_w = self.cnn_input_size_to_output_size(
            (self.args.image_height, fake_input_width))
        print('CNN out height %d, width %d' % (cnn_out_h, cnn_out_w))
        cnn_out_c = self.cnn_output_num_channels()

        cnn_feat_size = cnn_out_c * cnn_out_h

        print('CNN out height %d' % (cnn_out_h))
        print('CNN out channels %d' % (cnn_out_c))
        print('CNN feature size (channels %d x height %d) = %d' %
              (cnn_out_c, cnn_out_h, cnn_feat_size))

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.args.encoder_dim),
            nn.ReLU(inplace=True)
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_tokens, src_widths):
        if self.args.image_verbose:
            print('\nENCODER: forward input', src_tokens.shape)

        x = self.cnn(src_tokens)

        if self.args.image_verbose:
            print('ENCODER: forward features out', x.shape)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, c * h)
        x = self.bridge_layer(x)
        if self.args.image_verbose:
            print('ENCODER: forward bridge out', x.shape)
        x = x.view(w, b, -1)
        if self.args.image_verbose:
            print('ENCODER: forward bridge view', x.shape)

        actual_cnn_output_widths = [self.cnn_input_size_to_output_size((self.args.image_height, width))[1] for width in
                                    src_widths.data]

        # print('encoder actual_cnn_output_widths', actual_cnn_output_widths)
        return {
            'encoder_out': x,
            'encoder_actual_cnn_output_widths': actual_cnn_output_widths
        }

    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c

    def calculate_hw(self, module, out_h, out_w):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
            if isinstance(module.padding, tuple):
                padding_y, padding_x = module.padding
            else:
                padding_y = padding_x = module.padding
            if isinstance(module.dilation, tuple):
                dilation_y, dilation_x = module.dilation
            else:
                dilation_y = dilation_x = module.dilation
            if isinstance(module.stride, tuple):
                stride_y, stride_x = module.stride
            else:
                stride_y = stride_x = module.stride
            if isinstance(module.kernel_size, tuple):
                kernel_size_y, kernel_size_x = module.kernel_size
            else:
                kernel_size_y = kernel_size_x = module.kernel_size

            out_h = math.floor(
                (out_h + 2.0 * padding_y - dilation_y *
                 (kernel_size_y - 1) - 1) / stride_y + 1)
            out_w = math.floor(
                (out_w + 2.0 * padding_x - dilation_x *
                 (kernel_size_x - 1) - 1) / stride_x + 1)
        elif isinstance(module, nn.FractionalMaxPool2d):
            if module.output_size is not None:
                out_h, out_w = module.output_size
            else:
                rh, rw = module.output_ratio
                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)

        return out_h, out_w

    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size

        for module in self.cnn.modules():
            out_h, out_w = self.calculate_hw(module, out_h, out_w)

        return (out_h, out_w)

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]


class OCRCNNDecoder(torch.nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()

        self.args = args
        self.dictionary = dictionary

        if self.args.image_verbose:
            print('...LSTM input size', self.args.encoder_dim)

        self.lstm = nn.LSTM(self.args.encoder_dim, self.args.decoder_lstm_units, num_layers=self.args.decoder_lstm_layers,
                            dropout=self.args.decoder_lstm_dropout, bidirectional=True)

        self.prob_layer = nn.Sequential(
            nn.Linear(2 * self.args.decoder_lstm_units, len(self.dictionary))
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

        lstm_params = 0
        for param in self.lstm.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            lstm_params += local_params

        if self.args.image_verbose:
            print("\tLSTM Params = %d" % lstm_params)

            print("Model looks like:")
            print(repr(self))

        if torch.cuda.is_available():
            print('...using GPUS')
            self.lstm = self.lstm.cuda()
            self.prob_layer = self.prob_layer.cuda()
        else:
            print("Warning: Runnig model on CPU")

    def forward(self, encoder_output):

        encoder_out = encoder_output['encoder_out']
        encoder_widths = encoder_output['encoder_actual_cnn_output_widths']

        if self.args.image_verbose:
            print("\nDECODER: encoder output", encoder_out.shape)

        packed_lstm_input = rnn_pack(encoder_out, encoder_widths)
        packed_lstm_output, _ = self.lstm(packed_lstm_input)
        lstm_output, lstm_output_lengths = rnn_unpack(packed_lstm_output)

        if self.args.image_verbose:
            print("DECODER: lstm_output", lstm_output.shape)
            print("DECODER: lstm_output_lengths",
                  lstm_output_lengths.shape, lstm_output_lengths)

        w = lstm_output.size(0)

        seq_len, b, embed_dim = encoder_out.size()

        lstm_output_view = lstm_output.view(-1, lstm_output.size(2))
        prob_output = self.prob_layer(lstm_output_view)
        prob_output_view = prob_output.view(w, b, -1)

        if self.args.image_verbose:
            print("DECODER: lstm_output_view", lstm_output_view.shape)
            print("DECODER: prob_output", prob_output.shape)
            print("DECODER: prob_output_view", prob_output_view.shape)

        return prob_output_view, lstm_output_lengths.to(torch.int32)


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

    samples_train_output = args.output + '/samples_train'
    if not os.path.exists(samples_train_output):
        os.makedirs(samples_train_output)

    samples_valid_output = args.output + '/samples_valid'
    if not os.path.exists(samples_valid_output):
        os.makedirs(samples_valid_output)

    train_cache_output = args.output + '/train_cache_output'
    if not os.path.exists(train_cache_output):
        os.makedirs(train_cache_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.augment:
        train_transform = transforms.Compose([
            ImageAug(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    train_dataset = ImageDataset(text_file_path=args.train, font_file_path=args.train_font,
                                 train_cache_output=train_cache_output,
                                 max_image_cache=args.max_image_cache,
                                 save_images=args.save_images,
                                 max_seed=args.max_seed,
                                 max_cache_write=args.max_cache_write,
                                 image_height=args.image_height,
                                 use_image_cache=args.use_image_cache,
                                 transform=train_transform, image_verbose=args.image_verbose)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              sampler=GroupedSampler(
                                                  train_dataset, rand=True),
                                              collate_fn=lambda b: SortByWidthCollater(
                                                  b, args.image_verbose),
                                              num_workers=args.num_workers,
                                              pin_memory=True, drop_last=True)

    if args.image_verbose:
        print('Dictionary...')
        tgt_dict = train_dataset.alphabet
        for i in range(len(tgt_dict)):
            if i < 25:
                print(i, tgt_dict.idx_to_char[i])
            elif i > len(tgt_dict) - 25:
                print(i, tgt_dict.idx_to_char[i])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    valid_dataset = ImageDataset(text_file_path=args.valid, font_file_path=args.valid_font,
                                 image_height=args.image_height,
                                 transform=valid_transform,
                                 alphabet=train_dataset.alphabet)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                              sampler=GroupedSampler(
                                                  valid_dataset, rand=True),
                                              collate_fn=lambda b: SortByWidthCollater(
                                                  b, args.image_verbose),
                                              num_workers=0)

    encoder = OCRCNNEncoder(args)
    decoder = OCRCNNDecoder(args, train_dataset.alphabet)
    model = OCRCRNNModel(encoder, decoder, train_dataset.alphabet)
    model.to(device)

    best_cer = 9999

    last_checkpoint = os.path.join(args.output, 'checkpoints/model.pth')
    LOG.info('...searching for %s', last_checkpoint)
    if os.path.isfile(last_checkpoint):
        checkpoint = torch.load(last_checkpoint)
        LOG.info('Loading checkpoint...')
        LOG.info(' epoch %d', checkpoint['epoch'])
        LOG.info(' loss %f', checkpoint['loss'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # summary(encoder, input_size=(3, args.image_height, 800))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, min_lr=args.min_lr, verbose=True)

    model.train()

    for epoch in range(args.epochs):
        for i, sample in enumerate(trainloader):

            iteration_start = time.time()

            sample = move_to_cuda(sample)

            batch_size = len(sample['net_input']['src_widths'])

            optimizer.zero_grad()

            net_output, model_output_actual_lengths = model(
                **sample['net_input'])
            log_probs = F.log_softmax(net_output, dim=2)
            targets = sample['target']
            target_lengths = sample['target_length']
            # CUDA, PyTorch native implementation: OK
            # torch.backends.cudnn.enabled = False
            loss = F.ctc_loss(log_probs.to('cuda'), targets.to('cuda'), model_output_actual_lengths, target_lengths,
                              reduction='mean',
                              zero_infinity=True)

            loss.backward()
            optimizer.step()

            duration = time.time() - iteration_start
            examples_per_sec = args.batch_size / duration
            sec_per_batch = float(duration)

            if i % 50 == 0 and i > 0:
                LOG.info("Epoch: %d (%d/%d), Batch Size: %d, Group: %d, Loss: %.4f, LR: %.8f, ex/sec: %.1f, sec/batch: %.2f",
                         epoch, i + 1 % len(trainloader),
                         len(trainloader), batch_size, sample['group_id'],
                         loss.item(),
                         get_lr(optimizer), examples_per_sec, sec_per_batch)

            if epoch % 10 == 0 and i % 1000 == 0:
                image_list = sample['net_input']['src_tokens'].cpu().numpy()
                label_list = sample['seed_text']
                for img_idx, img in enumerate(image_list):
                    seed_text = label_list[img_idx]

                    image = np.uint8(img.transpose((1, 2, 0)) * 255)
                    image = image[:, :, ::-1].copy()  # rgb to bgr
                    (h, w) = image.shape[:2]

                    outpath = samples_train_output + '/' + \
                        'epoch' + str(epoch) + '_batch' + str(i) + '_h' + str(h) + '_w' + \
                        str(w) + '_' + seed_text + '.png'
                    cv2.imwrite(outpath, image)

        # for each batch
        cer_running_avg = 0
        wer_running_avg = 0
        n_samples = 0
        loss_running_avg = 0
        total_val_images = 0

        model.eval()
        with torch.no_grad():
            for sample_ctr, sample in enumerate(validloader):

                sample = move_to_cuda(sample)

                batch_size = len(sample['net_input']['src_widths'])
                total_val_images += batch_size

                net_output, model_output_actual_lengths = model(
                    **sample['net_input'])
                log_probs = F.log_softmax(net_output, dim=2)
                targets = sample['target']
                target_lengths = sample['target_length']
                # CUDA, PyTorch native implementation: OK
                # torch.backends.cudnn.enabled = False
                loss = F.ctc_loss(log_probs.to('cuda'), targets.to('cuda'), model_output_actual_lengths, target_lengths,
                                  reduction='mean',
                                  zero_infinity=True)

                n_samples += 1

                loss_running_avg += (loss -
                                     loss_running_avg) / n_samples

                hyp_transcriptions = model.decode(
                    net_output, model_output_actual_lengths)

                batch_cer = 0
                batch_wer = 0

                for hyp_ctr in range(len(hyp_transcriptions)):
                    utf8_ref_text = sample['seed_text'][hyp_ctr]
                    uxxxx_ref_text = utf8_to_uxxxx(utf8_ref_text)

                    utf8_hyp_text = uxxxx_to_utf8(hyp_transcriptions[hyp_ctr])
                    uxxxx_hyp_text = hyp_transcriptions[hyp_ctr]

                    cer, wer = compute_cer_wer(
                        uxxxx_hyp_text, uxxxx_ref_text)

                    batch_cer += cer
                    batch_wer += wer

                    if sample_ctr % 10 == 0 and hyp_ctr % 20 == 0:
                        print('\nhyp', utf8_hyp_text, uxxxx_hyp_text)
                        print('ref', utf8_ref_text, uxxxx_ref_text)
                        print('cer %f \twer %f' % (cer, wer))

                cer_running_avg += (batch_cer / batch_size -
                                    cer_running_avg) / n_samples
                wer_running_avg += (batch_wer / batch_size -
                                    wer_running_avg) / n_samples

                if epoch % 10 == 0 and sample_ctr % 100 == 0:
                    image_list = sample['net_input']['src_tokens'].cpu(
                    ).numpy()
                    label_list = sample['seed_text']
                    for img_idx, img in enumerate(image_list):
                        seed_text = label_list[img_idx]

                        image = np.uint8(img.transpose((1, 2, 0)) * 255)
                        image = image[:, :, ::-1].copy()  # rgb to bgr
                        (h, w) = image.shape[:2]

                        outpath = samples_valid_output + '/' + \
                            'epoch' + str(epoch) + '_batch' + str(sample_ctr) + '_h' + str(h) + '_w' + \
                            str(w) + '_' + seed_text + '.png'
                        cv2.imwrite(outpath, image)

        print("Validation Images: %d Loss: %f \tVal CER: %f\tVal WER: %f \tSamples: %d" %
              (total_val_images, loss_running_avg, cer_running_avg, wer_running_avg, n_samples))

        if best_cer is None:
            best_cer = cer_running_avg
        elif cer_running_avg < best_cer:
            best_cer = cer_running_avg

            torch.save({'epoch': epoch,
                        'loss': loss.item(),
                        'cer': cer_running_avg,
                        'wer': wer_running_avg,
                        'alphabet': train_dataset.alphabet,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        },
                       os.path.join(ckpts_output, 'model_ckpt_best.pth'))

        model.train()

        scheduler.step(cer_running_avg)

        torch.save({'epoch': epoch,
                    'loss': loss.item(),
                    'cer': cer_running_avg,
                    'wer': wer_running_avg,
                    'alphabet': train_dataset.alphabet,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    },
                   os.path.join(ckpts_output, 'model_ckpt_last.pth'))

        print("Validation Loss: %f \tVal CER: %f\tVal WER: %f \tSamples: %d" %
              (loss_running_avg, cer_running_avg, wer_running_avg, n_samples))

    LOG.info('...complete, time %s', time.clock() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
