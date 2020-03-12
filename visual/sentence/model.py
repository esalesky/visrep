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
import torch.nn.functional as F
import logging
from text_utils import compute_cer_wer, uxxxx_to_utf8, utf8_to_uxxxx
from torch.nn.utils.rnn import pack_padded_sequence as rnn_pack
from torch.nn.utils.rnn import pad_packed_sequence as rnn_unpack
from models.resnet import resnet18
from models.vista import vista_fractional, vista_maxpool


LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


class OcrModel(torch.nn.Module):
    def __init__(self, args, encoder, decoder, dictionary):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self.encoder = encoder
        self.decoder = decoder

        if self.args.image_verbose:
            print("Model looks like:")
            print(repr(self))

    def forward(self, src_tokens, src_widths):
        encoder_out = self.encoder(src_tokens, src_widths)
        decoder_out = self.decoder(encoder_out)

        return decoder_out

    def decode(self, model_output, batch_actual_timesteps):
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


class OcrEncoder(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        print('cnn', args.encoder_arch)

        # Fixed line height of 32
        if self.args.encoder_arch.startswith('vista_fractional'):
            cnn_out_c = 256
            cnn_out_h = 8
            self.cnn = vista_fractional()
        elif self.args.encoder_arch.startswith('vista_maxpool'):
            cnn_out_c = 256
            cnn_out_h = 8
            self.cnn = vista_maxpool()
        elif self.args.encoder_arch.startswith('resnet18'):
            cnn_out_c = 512  # 56  # 512
            cnn_out_h = 4  # 8  # 4
            self.cnn = resnet18()

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

        x_cnn = self.cnn(src_tokens)

        if self.args.image_verbose:
            print('ENCODER: forward features out', x_cnn.shape)

        b, c, h, w = x_cnn.size()
        x = x_cnn.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, c * h)

        x = self.bridge_layer(x)
        if self.args.image_verbose:
            print('ENCODER: forward bridge out', x.shape)

        x = x.view(w, b, -1)
        if self.args.image_verbose:
            print('ENCODER: forward bridge view', x.shape)

        actual_cnn_output_widths = [self.cnn_input_size_to_output_size((self.args.image_height, width))[1] for width in
                                    src_widths.data]

        return {
            'encoder_out': x,
            'encoder_cnn_shape': list(x_cnn.shape),
            'input_shape': list(src_tokens.shape),
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

            if kernel_size_x > 1:  # resnet18 fix
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


class OcrDecoder(torch.nn.Module):
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

        w = lstm_output.size(0)

        seq_len, b, embed_dim = encoder_out.size()

        lstm_output_view = lstm_output.view(-1, lstm_output.size(2))
        prob_output = self.prob_layer(lstm_output_view)
        prob_output_view = prob_output.view(w, b, -1)

        out_meta = {
            'input_shape': encoder_output['input_shape'],
            'cnn_shape': encoder_output['encoder_cnn_shape'],
            'bridge_shape': list(encoder_out.shape),
            'encoder_widths_len': len(encoder_widths),
            'lstm_output_shape': list(lstm_output.shape),
            'lstm_output': lstm_output,
            'lstm_output_view_shape': list(lstm_output_view.shape),
            'prob_output_shape': list(prob_output.shape),
            'prob_output_view_shape': list(prob_output_view.shape),
            'prob_output_view': prob_output_view,
            'lstm_output_lengths': lstm_output_lengths.to(torch.int32)
        }
        if self.args.image_verbose:
            print("DECODER: lstm_output", lstm_output.shape)
            print("DECODER: lstm_output_view", lstm_output_view.shape)
            print("DECODER: prob_output", prob_output.shape)
            print("DECODER: prob_output_view", prob_output_view.shape)

        return out_meta
