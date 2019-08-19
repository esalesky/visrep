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

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, FairseqVisualModel, register_model,
    register_model_architecture, FLCEncoder, OldFLCEncoder, VisualEncoder, VisualEdgeEncoder, CharCNNEncoder,
    MultiFeatEncoder, ImageWordEncoder
)

from fairseq.models.transformer import Embedding, RobustTransformerEncoder, TransformerEncoder, TransformerDecoder

import numpy as np
import cv2


@register_model('visual_transformer')
class VisualTransformerModel(FairseqVisualModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder, visual_encoder):
        super().__init__(encoder, decoder, visual_encoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        parser.add_argument('--robust-embedder-type', type=str,
                            action='store', default='None',
                            help='will create Robust bag of chars encoder')
        parser.add_argument('--robust-embedder-resource', type=str,
                            action='store', default=None,
                            help='path to torch tensor with img embeddings')
        parser.add_argument('--edge-threshold', type=float,
                            action='store', default=0.0,
                            help='controls the softness of edge detection')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # self.args = args

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if args.num_source_feats > 1:
            encoder = RobustTransformerEncoder(args, src_dict, encoder_embed_tokens,
                                               num_source_feats=args.num_source_feats,
                                               robust_embedder_type=args.robust_embedder_type,
                                               robust_embedder_resource=args.robust_embedder_resource,
                                               edge_threshold=args.edge_threshold)
        else:
            encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)

        visual_encoder = None
        if args.image_type is not None:
            visual_encoder = ImageWordEncoder(
                dictionary=task.source_dictionary,
                input_channels=args.image_channels,
                input_line_height=args.image_height,
                input_line_width=args.image_width,
                maxpool_ratio_height=args.image_maxpool_height,
                maxpool_ratio_width=args.image_maxpool_width,
                kernel_size_2d=args.image_kernel,
                stride_2d=args.image_stride,
                padding_2d=args.image_pad,
                output_dim=args.image_embed_dim)
            
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

        return VisualTransformerModel(args, encoder, decoder, visual_encoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_images=None):
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

        # print('...visual_transformer forward')
        if self.args.image_verbose:
            # print('...visual_transformer forward', src_images.shape)
            # print('......first image', src_images[0][0].shape, len(src_images[0]))

            b, t, c, h, w = src_images.shape
            # print(b, t, c, h, w)
            for ctr in range(len(src_images[0])):
                token_id = src_tokens[0][ctr].cpu()
                # print(int(token_id))
                word = self.encoder.dictionary.__getitem__(int(token_id))
                # print(int(token_id), word)
                image = np.uint8(src_images[0][ctr].cpu().squeeze()).transpose((1, 2, 0)) * 255
                outimage = self.args.image_samples_path + '/word_' + str(int(token_id)) + '_' + str(word) + '.png'
                # print('write image ...', outimage)
                cv2.imwrite(outimage, image)

            # src_images = src_images.contiguous().view(-1, src_images.size(-3), src_images.size(-2), src_images.size(-1))
            # print('...image shape', src_images.shape)
            src_images = src_images.view(-1, src_images.size(-3), src_images.size(-2), src_images.size(-1))
            # print('...image reshape', src_images.shape)
            # src_images = src_images.view(b, t, c, h, w)
            # print('...image back', src_images.shape)

        if src_images is not None:
            visual_encoder_out = self.visual_encoder(src_images)
            # print('visual encoder out shape', visual_encoder_out['encoder_out'].shape)
            visual_encoder_out['encoder_out'] = visual_encoder_out['encoder_out'].view(b, t, self.args.image_embed_dim)
            # print('visual encoder out RESHAPE', visual_encoder_out['encoder_out'].shape)

        # cv2.imwrite(self.image_samples_path + '/word_' + src_word[idx] + '_' + str(index) + '_' + str(idx) + '.png', img)
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        # decoder_out['source_embedding_out'] = encoder_out['embedding_out']
        return decoder_out


@register_model_architecture('visual_transformer', 'visual_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('visual_transformer', 'visual_transformer_iwslt_de_en')
def visual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    base_architecture(args)

