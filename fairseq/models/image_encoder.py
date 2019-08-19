
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import math
import logging

logger = logging.getLogger('root')

from . import (
    FairseqEncoder,
)

'''

Image word encoder

CNN input c 3 h 30 w 120 
CNN out c 256, h 7, w 58
CNN feature size 1792 (c * h * w)
CNN output dim 512

'''

class ImageWordEncoder(FairseqEncoder):

    def __init__(self, dictionary, input_channels=3, input_line_height=30, 
                 input_line_width=200, maxpool_ratio_height = 0.5, 
                 maxpool_ratio_width=0.7, kernel_size_2d=3, stride_2d=1, 
                 padding_2d=1, output_dim=512):
        super().__init__(dictionary)
        
        self.output_dim = output_dim # 512
        self.num_in_channels = input_channels # 3
        self.input_line_height = input_line_height
        self.input_line_width = input_line_width
        self.kernel_size_2d = kernel_size_2d
        self.stride_2d = stride_2d
        self.padding_2d = padding_2d
        self.maxpool_ratio_height = maxpool_ratio_height
        self.maxpool_ratio_width = maxpool_ratio_width


        self.encoder = nn.Sequential(
            *self.ConvBNReLU(
                self.num_in_channels, 64, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),
            *self.ConvBNReLU(
                64, 64, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),

            nn.FractionalMaxPool2d(
                2, output_ratio=(self.maxpool_ratio_height,
                                 self.maxpool_ratio_width)),

            *self.ConvBNReLU(
                64, 128, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),
            *self.ConvBNReLU(
                128, 128, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),

            nn.FractionalMaxPool2d(
                2, output_ratio=(self.maxpool_ratio_height,
                                 self.maxpool_ratio_width)),

            *self.ConvBNReLU(
                128, 256, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),
            *self.ConvBNReLU(
                256, 256, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d),
            *self.ConvBNReLU(
                256, 256, kernel_size_2d=self.kernel_size_2d,
                stride_2d=self.stride_2d, padding_2d=self.padding_2d)
        )
        

        cnn_out_h, cnn_out_w = self.cnn_input_size_to_output_size(
            (self.input_line_height, self.input_line_width))
        cnn_out_c = self.cnn_output_num_channels()

        cnn_feat_size = cnn_out_c * cnn_out_h * cnn_out_w

        logger.info('CNN input c %d h %d w %d ' % (self.num_in_channels, self.input_line_height, self.input_line_width ))
        logger.info('CNN output c %d h %d w %d' % (cnn_out_c, cnn_out_h, cnn_out_w))
        logger.info('CNN feature size %d (c*h*w)' % (cnn_feat_size))
        logger.info('CNN output dim %d' % (self.output_dim))
        
        self.encode_fc = nn.Sequential(
            nn.Linear(cnn_feat_size, self.output_dim),
            nn.ReLU(inplace=True)
        )

    def ConvBNReLU(self, nInputMaps, nOutputMaps, kernel_size_2d=3,
                   stride_2d=1, padding_2d=1):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=kernel_size_2d, 
                          padding=padding_2d, stride=stride_2d),
                          nn.BatchNorm2d(nOutputMaps), nn.ReLU(inplace=True)]
        
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

    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c

    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size

        for module in self.encoder.modules():
            out_h, out_w = self.calculate_hw(module, out_h, out_w)

        return (out_h, out_w)

    def forward(self, src_images):
        encoded = self.encoder(src_images)
        batch, channel, height, width = encoded.size()
        encode_fc_view = encoded.view(-1, channel * height * width)
        encode_fc = self.encode_fc(encode_fc_view)
        return {
            'encoder_out': encode_fc,  
            'encoder_padding_mask': None,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        #return self.embed_positions.max_positions()
        return int(1e5)  # an arbitrary large number

