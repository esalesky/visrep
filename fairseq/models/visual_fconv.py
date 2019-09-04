# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)

from .image_encoder import ImageWordEncoder

from . import (
    VisualFairseqEncoder, FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, FairseqVisualModel,
    FairseqLanguageModel, register_model, register_model_architecture,
)

from .fconv import (
    FConvEncoder, AttentionLayer, FConvDecoder, extend_conv_spec, Embedding, PositionalEmbedding,
    Linear, LinearizedConv1d, ConvTBC,
)

import numpy as np
import cv2
import os


@register_model('visual_fconv')
class VisualFConvModel(FairseqVisualModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        if not os.path.exists(args.image_samples_path):
            os.makedirs(args.image_samples_path)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        encoder = VisualWordFConvEncoder(
            dictionary=task.source_dictionary,
            input_channels=args.image_channels,
            input_line_height=args.image_height,
            input_line_width=args.image_width,
            maxpool_ratio_height=args.image_maxpool_height,
            maxpool_ratio_width=args.image_maxpool_width,
            kernel_size_2d=args.image_kernel,
            stride_2d=args.image_stride,
            padding_2d=args.image_pad,
            embed_dict=encoder_embed_dict,
            output_dim=args.image_embed_dim,
            dropout=args.dropout,
            max_positions=args.max_source_positions)

        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
        )
        return VisualFConvModel(args, encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_images):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """

        if self.args.image_verbose:
            b, t, c, h, w = src_images.shape
            for ctr in range(len(src_images[0])):
                token_id = src_tokens[0][ctr].cpu()
                word = self.encoder.dictionary.__getitem__(int(token_id))
                image = np.uint8(src_images[0][ctr].cpu().squeeze()).transpose((1, 2, 0)) * 255
                outimage = self.args.image_samples_path + '/word_' + str(int(token_id)) + '_' + str(word) + '.png'
                cv2.imwrite(outimage, image)

        visual_encoder_out = self.encoder(src_images, src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, visual_encoder_out)

        return decoder_out


class VisualWordFConvEncoder(VisualFairseqEncoder):

    def __init__(self, dictionary, input_channels=3, input_line_height=30,
                 input_line_width=200, maxpool_ratio_height=0.5,
                 maxpool_ratio_width=0.7, kernel_size_2d=3, stride_2d=1,
                 padding_2d=1, output_dim=512,
                 embed_dict=None, dropout=0.1, left_pad=True, max_positions=1024):
        super().__init__(dictionary)

        self.output_dim = output_dim
        self.dropout = dropout
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, output_dim, self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, dictionary, self.embed_tokens)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            output_dim,
            self.padding_idx,
            left_pad=left_pad,
        )

        self.visual_encoder = ImageWordEncoder(
            dictionary=dictionary,
            input_channels=input_channels,
            input_line_height=input_line_height,
            input_line_width=input_line_width,
            maxpool_ratio_height=maxpool_ratio_height,
            maxpool_ratio_width=maxpool_ratio_width,
            kernel_size_2d=kernel_size_2d,
            stride_2d=stride_2d,
            padding_2d=padding_2d,
            output_dim=output_dim
            )

    def forward(self, src_images, src_tokens, src_lengths):
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        b, t, c, h, w = src_images.shape
        # print('VisualWordFConvEncoder forward input b {} t {} c {} h {} w {} '.format(b, t, c, h, w))
        encode_fc = self.visual_encoder(src_images)
        # print('VisualWordFConvEncoder forward output {}'.format(encode_fc.shape))
        encode_fc_y = (encode_fc + input_embedding) * math.sqrt(0.5)
        encode_fc_y = encode_fc_y.view(b, t, self.output_dim)

        return {
            'encoder_out': (encode_fc, encode_fc_y),
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


@register_model_architecture('visual_fconv', 'visual_fconv')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('visual_fconv', 'visual_fconv_iwslt_de_en')
def fconv_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)

