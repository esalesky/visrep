import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)


class AlignOCR(nn.Module):
    """Vista OCR """

    def __init__(self):
        """
        """

        super(AlignOCR, self).__init__()

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(1, 64),
            *self.ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2),
            *self.ConvBNReLU(64, 128),
            *self.ConvBNReLU(128, 128),
            nn.MaxPool2d(2, 2),
            *self.ConvBNReLU(128, 256),
            *self.ConvBNReLU(256, 256),
            *self.ConvBNReLU(256, 256)
        )

    def forward(self, x):
        """
        Takes a tensor of size (batch x frames x channels x width x height).
        Runs convolutions on every frame. The CNN will reduce every frame
        to dims (256 x w' x h').
        """

        # shape is (batch x token x (channel) x height x width)
        # if channel is 1 it may be collapsed
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # logger.info("VIEW", x.shape)
        x = self.cnn(x)

        return x

    def ConvBNReLU(self, nInputMaps, nOutputMaps, stride=1):
        return [nn.Conv2d(nInputMaps, nOutputMaps, stride=stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

class AlignOcrEncoder(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # self.args.image_verbose = True

        cnn_out_c = 256
        cnn_out_h = 8

        self.cnn = align_ocr()

        cnn_feat_size = cnn_out_c * cnn_out_h

        logger.info('CNN out height %d', cnn_out_h)
        logger.info('CNN out channels %d', cnn_out_c)
        logger.info('CNN feature size (channels %d x height %d) = %d',
                 cnn_out_c, cnn_out_h, cnn_feat_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 1))

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.args.encoder_embed_dim),
            nn.ReLU(inplace=True)
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_tokens):  # , src_widths):
        """
        input: (batch, len, channels, height, width)
        output: (batch, len, embed_size)
        """

        logger.debug('ENCODER: forward input %s', src_tokens.shape)

        batch_size, src_len, *image_dims = src_tokens.shape
        src_tokens = src_tokens.view(batch_size * src_len, *image_dims)

        logger.debug('ENCODER: collapse tokens %s', src_tokens.shape)

        x_cnn = self.cnn(src_tokens)

        logger.debug('ENCODER: forward cnn features out (b, c, h, w) %s', x_cnn.shape)

        x = self.avg_pool(x_cnn)
        logger.debug('ENCODER: avg pool %s', x.shape)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        logger.debug('ENCODER: permute (w, b, c, h) %s', x.shape)

        x = x.view(-1, c * h)
        logger.debug('ENCODER: view (w*b, c*h) %s', x.shape)

        x = self.bridge_layer(x)
        logger.debug('ENCODER: forward bridge out %s', x.shape)

        x = x.view(batch_size, src_len, -1)
        logger.debug('ENCODER: forward bridge view %s', x.shape)

        return {
            'encoder_out': x,
            'encoder_cnn_shape': list(x_cnn.shape),
            'input_shape': list(src_tokens.shape),
        }

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

def align_ocr():
    model = AlignOCR()
    return model
