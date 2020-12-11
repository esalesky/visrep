import torch
import torch.nn as nn

import logging
LOG = logging.getLogger(__name__)


class AlignOCR(nn.Module):
    """Vista OCR """

    def __init__(self):
        super(AlignOCR, self).__init__()

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(1, 64),
            nn.MaxPool2d(2, 2),
            *self.ConvBNReLU(64, 128),
            nn.MaxPool2d(2, 2),
            *self.ConvBNReLU(128, 256)
        )

    def forward(self, x):

        b, t, c, h, w = x.shape  # batch x token x channel x height x width
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.cnn(x)

        return x

    def ConvBNReLU(self, nInputMaps, nOutputMaps, stride=1):
        return [nn.Conv2d(nInputMaps, nOutputMaps, stride=stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]


class AlignOcrModel(torch.nn.Module):
    def __init__(self, args, vocab, eval_only=False):
        super().__init__()
        self.args = args

        self.eval_only = eval_only

        self.vocab = vocab
        self.encoder = AlignOcrEncoder(args)
        self.decoder = AlignOcrDecoder(args, vocab)

        LOG.info('AlignOcrModel eval_only %s', self.eval_only)
        LOG.info(repr(self))

    def forward(self, src_tokens):  # , src_widths):
        encoder_out = self.encoder(src_tokens)  # , src_widths)
        decoder_out = self.decoder(encoder_out)

        return decoder_out  # decoder_out

    def train(self, mode=True):

        if self.eval_only:
            mode = False

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


class AlignOcrEncoder(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # self.args.image_verbose = True

        cnn_out_c = 256
        cnn_out_h = 8

        self.cnn = align_ocr()

        cnn_feat_size = cnn_out_c * cnn_out_h

        LOG.info('CNN out height %d', cnn_out_h)
        LOG.info('CNN out channels %d', cnn_out_c)
        LOG.info('CNN feature size (channels %d x height %d) = %d',
                 cnn_out_c, cnn_out_h, cnn_feat_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 1))

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.args.image_embed_dim),
            nn.ReLU(inplace=True)
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_tokens):  # , src_widths):

        LOG.debug('ENCODER: forward input %s', src_tokens.shape)

        x_cnn = self.cnn(src_tokens)

        LOG.debug('ENCODER: forward cnn features out (b, c, h, w) %s', x_cnn.shape)

        x = self.avg_pool(x_cnn)
        LOG.debug('ENCODER: avg pool %s', x.shape)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        LOG.debug('ENCODER: permute (w, b, c, h) %s', x.shape)

        x = x.view(-1, c * h)
        LOG.debug('ENCODER: view (w*b, c*h) %s', x.shape)

        x = self.bridge_layer(x)
        LOG.debug('ENCODER: forward bridge out %s', x.shape)

        x = x.view(w, b, -1)
        LOG.debug('ENCODER: forward bridge view %s', x.shape)

        return {
            'encoder_out': x,
            'encoder_cnn_shape': list(x_cnn.shape),
            'input_shape': list(src_tokens.shape),
        }

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]


class AlignOcrDecoder(torch.nn.Module):
    def __init__(self, args, vocab):
        super().__init__()

        self.args = args
        self.vocab = vocab

        self.classifier = nn.Sequential(
            nn.Linear(self.args.image_embed_dim, len(self.vocab))
        )

    def forward(self, encoder_output):
        embeddings = encoder_output['encoder_out'].squeeze()
        LOG.debug('DECODER: embeddings %s', embeddings.shape)

        logits = self.classifier(embeddings)

        LOG.debug('DECODER: logits %s', logits.shape)

        out_meta = {
            'input_shape': encoder_output['input_shape'],
            'encoder_cnn_shape': encoder_output['encoder_cnn_shape'],
            'embeddings': embeddings,
            'logits': logits,
        }

        return out_meta


def align_ocr():
    model = AlignOCR()
    return model
