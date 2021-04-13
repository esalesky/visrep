import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)


class AlignOcrEncoder(torch.nn.Module):
    """
    An encoder is the visual CNN, followed by average pooling, then
    a bridge layer to the embedding size.
    """

    def ConvBNReLU(self, nInputMaps, nOutputMaps, stride=1, kernel_size=3, padding=1):
        return [nn.Conv2d(nInputMaps, nOutputMaps, stride=stride, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

    def __init__(self, args):
        super().__init__()
        self.args = args

        # self.args.image_verbose = True

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

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 1))

        cnn_out_c = 256
        cnn_out_h = 8
        cnn_feat_size = cnn_out_c * cnn_out_h

        logger.info('CNN out height %d', cnn_out_h)
        logger.info('CNN out channels %d', cnn_out_c)
        logger.info('CNN feature size (channels %d x height %d) = %d',
                 cnn_out_c, cnn_out_h, cnn_feat_size)

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.args.encoder_embed_dim),
            nn.ReLU(inplace=True)
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_tokens):  # , src_widths):
        """
        Takes batch input with shape (batch, len, channels, height, width)
        and returns embeddings for all batch items, with shape (batch, len, embed_size).
        """
        logger.debug('ENCODER: forward input %s', src_tokens.shape)

        batch_size, src_len, *image_dims = src_tokens.shape
        src_tokens = src_tokens.view(batch_size * src_len, *image_dims)

        logger.debug('ENCODER: collapse tokens %s', src_tokens.shape)

        ## APPLY THE CNN
        x_cnn = self.cnn(src_tokens)

        logger.debug('ENCODER: forward cnn features out (b, c, h, w) %s', x_cnn.shape)

        ## AVERAGE POOLING
        x = self.avg_pool(x_cnn)
        logger.debug('ENCODER: avg pool %s', x.shape)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        logger.debug('ENCODER: permute (w, b, c, h) %s', x.shape)

        x = x.view(-1, c * h)
        logger.debug('ENCODER: view (w*b, c*h) %s', x.shape)

        ## APPLY THE BRIDGE LAYER
        x = self.bridge_layer(x)
        logger.debug('ENCODER: forward bridge out %s', x.shape)

        x = x.view(batch_size, src_len, -1)
        logger.debug('ENCODER: forward bridge view %s', x.shape)

        return x
