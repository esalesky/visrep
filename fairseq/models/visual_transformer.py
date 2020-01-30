import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import os
import numpy as np

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import (
    TransformerEncoder,
    Embedding,
    Linear,
    TransformerDecoder,
)

from fairseq.modules import (
    VisualNet,
    VistaOCR,
    Softmax,
    VisualTrainer,
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('visualtransformer')
class VisualTransformerModel(BaseFairseqModel):
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

    # def __init__(self, args, visual_encoder, encoder, decoder):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        #self.visual_encoder = visual_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
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
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # fmt: on

        parser.add_argument('--freeze-encoder-embed', default=False, action='store_true',
                            help='Freeze the encoder embeddings')

    def forward(self, src_tokens, src_lengths, src_images, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, src_images=src_images, **kwargs)

        if self.args.image_verbose:
            print('ENCODER: output (token, batch, embed_dim) %s' %
                  (str(encoder_out['encoder_out'].shape)))

        # 'encoder_out': x,  # T x B x C
        # 'encoder_padding_mask': encoder_padding_mask,  # B x T
        # 'encoder_embedding': encoder_embedding,  # B x T x C
        # 'encoder_states': encoder_states,  # List[T x B x C]
        # 'vis_encoder_out': vis_encoder_out,  # B, T, D embed_dim
        # 'visual_prelogits': visual_prelogits,

        if self.args.image_verbose:
            print('DECODER: input (encoder out) %s, prev decoder out (batch, tgt_len) %s' %
                  (str(encoder_out['encoder_out'].shape), str(prev_output_tokens.shape)))

        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs)

        if self.args.image_verbose:
            print('DECODER: output (batch, tgt len, vocab) %s' %
                  (str(decoder_out[0].shape)))

        if not self.args.image_disable:
            visual_prelogits = encoder_out['visual_prelogits']
        else:
            visual_prelogits = None

        return decoder_out, visual_prelogits

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def get_src_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['net_input']['src_tokens']

    def get_src_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            decoder_probs = self.decoder.get_normalized_probs(
                net_output, log_probs, sample)
            return decoder_probs
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)

    def extract_features(self, src_tokens, src_lengths, src_images, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out, _ = self.encoder(
            src_tokens, src_lengths=src_lengths, src_images=src_images, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

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
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embeddings not compatible with --decoder-embed-path')
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

        if args.image_disable:
            if args.image_embed_path:
                num_embeddings = len(src_dict)
                padding_idx = src_dict.pad()
                visual_embed_tokens = Embedding(
                    num_embeddings, args.encoder_embed_dim, padding_idx)
                visual_embed_dict = utils.parse_embedding(
                    args.image_embed_path)
                utils.load_embedding(
                    visual_embed_dict, src_dict, visual_embed_tokens)
                print('VIS_TRANSFORMER: use pretrained visual embeddings')
            else:
                visual_embed_tokens = None
        else:
            visual_embed_tokens = None

        # print(args)
        if args.image_verbose:
            print('VIS_TRANSFORMER: src_dict len %d' % (len(src_dict)))
            print('VIS_TRANSFORMER: tgt_dict len %d' % (len(tgt_dict)))

        if args.freeze_encoder_embed:
            print('...FREEZE encoder embed')
            encoder_embed_tokens.weight.requires_grad = False

        if visual_embed_tokens:
            if args.image_freeze_encoder_embed:
                print('VISUAL ENCODER: freeze pretrain visual_embed_tokens')
                visual_embed_tokens.weight.requires_grad = False

        #visual_encoder = cls.build_visual_encoder(args, src_dict)
        encoder = cls.build_encoder(
            args, src_dict, encoder_embed_tokens, visual_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        # return cls(args, visual_encoder, encoder, decoder)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, visual_embed_tokens):
        if not args.image_disable:
            if args.image_verbose:
                print('VIS_TRANSFORMER: setup TransformerEncoder')
            # return TransformerEncoder(args, src_dict, embed_tokens)
            visual_encoder = cls.build_visual_encoder(args, src_dict)
        else:
            visual_encoder = None
        return VisualTransformerEncoder(args, src_dict, embed_tokens, visual_embed_tokens, visual_encoder)

    @classmethod
    def build_visual_encoder(cls, args, src_dict):
        if args.image_verbose:
            print('VIS_TRANSFORMER: setup VisualNet')

        if 'vista' in args.image_backbone:
            backbone = VistaOCR(use_bridge=args.image_use_bridge, encoder_dim=args.image_embed_dim,
                                input_line_height=args.image_height, image_verbose=args.image_verbose)
        else:
            backbone = VisualNet(dim=512, input_shape=(args.image_height, args.image_width),
                                 model_name=args.image_backbone, extract=args.image_layer, normalize=False)

        # Get src normal prob with handle log_softmax
        head = Softmax(dim=512, dim_out=len(src_dict), log_softmax=False)
        return VisualTrainer(backbone, head)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.image_verbose:
            print('VIS_TRANSFORMER: setup TransformerDecoder')
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class VisualTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, visual_embed_tokens, visual_encoder):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.visual_embed_tokens = visual_embed_tokens
        if self.visual_embed_tokens == None:
            print('VISUAL ENCODER: pretrain visual_embed_tokens set to NONE')
        else:
            print('VISUAL ENCODER: using pretrain visual_embed_tokens')

        self.visual_encoder = visual_encoder
        if self.visual_encoder == None:
            print('VISUAL ENCODER: visual_encoder set to NONE')

        #self.print_ctr = 0
        self.args = args
        self.dropout = args.dropout

        if self.args.image_embed_type == 'concat':
            self.vis_linear = torch.nn.Linear(
                self.args.image_embed_dim * 2, self.args.image_embed_dim)
        else:
            self.vis_linear = torch.nn.Linear(
                self.args.image_embed_dim, self.args.image_embed_dim)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(
            args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward_visual_embedding(self, src_tokens, src_images):
        if not self.args.image_disable:
            b, t, c, h, w = src_images.shape  # batch x token x channel x height x width

            src_images = src_images.view(-1, src_images.size(-3),
                                         src_images.size(-2), src_images.size(-1))

            if self.args.image_verbose:
                print('ENCODER: input ((batch x token), channel, height, width) %s' % (
                    str(src_images.shape)))

            visual_out, visual_prelogits = self.visual_encoder(
                src_images)  # (B X T) X D embed_dim

            vis_encoder_out = visual_out.view(
                b, t, 512)  # batch, token, embed_dim

        else:
            if self.args.image_embed_path:
                vis_encoder_out = self.embed_scale * \
                    self.visual_embed_tokens(src_tokens)
                visual_prelogits = None
            else:
                vis_encoder_out = None
                visual_prelogits = None

        return vis_encoder_out, visual_prelogits

    def forward(self, src_tokens, src_lengths, src_images, cls_input=None, return_all_hiddens=False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        if self.args.image_verbose:
            print('ENCODER: input src_tokens %s, src_lengths %s' % (
                src_tokens.shape, src_lengths.shape))
            print('ENCODER: src_lengths %s' % (
                src_lengths))
        vis_encoder_out, visual_prelogits = self.forward_visual_embedding(
            src_tokens, src_images)

        if self.args.image_verbose:
            if type(vis_encoder_out) == type(None):
                print('ENCODER: visual embed NONE')
            else:
                print('ENCODER: visual embed ((batch x token), embed_dim) %s' % (
                    str(vis_encoder_out.shape)))

            if type(visual_prelogits) == type(None):
                print('ENCODER: visual embed prelogits NONE')
            else:
                print('ENCODER: visual embed prelogits %s' % (
                    str(visual_prelogits.shape)))

        x, encoder_embedding = self.forward_embedding(src_tokens)

        if self.args.image_verbose:
            print('ENCODER: token embed %s' %
                  str(x.shape))

        # if not self.args.image_disable:
        if type(vis_encoder_out) != type(None):
            if self.args.image_embed_type == 'concat':
                if self.args.image_verbose:
                    print('ENCODER: CONCAT input for tok and visual embed, tok %s, visual %s' %
                          (str(x.shape), str(vis_encoder_out.shape)))
                x_cat = torch.cat((x, vis_encoder_out), dim=2)
                if self.args.image_verbose:
                    print('ENCODER: CONCAT out %s' %
                          (str(x_cat.shape)))
                x = self.vis_linear(x_cat)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.args.image_verbose:
                    print('ENCODER: CONCAT after linear %s' %
                          (str(x.shape)))
            elif self.args.image_embed_type == 'avg':
                x_avg = torch.mean((x, vis_encoder_out), dim=2)
                x = self.vis_linear(x_avg)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.args.image_verbose:
                    print('ENCODER: AVG tok and visual embed, avg %s, out %s' %
                          (str(x_avg.shape), str(x.shape)))
            elif self.args.image_embed_type == 'visonly':
                x = vis_encoder_out
                x = self.vis_linear(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.args.image_verbose:
                    print('ENCODER: ONLY visual embed')
            else:  # self.args.image_embed_type == 'ignore':
                if self.args.image_verbose:
                    print('ENCODER: IGNORE visual embed')

            # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        encoder_out_dict = {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
            'vis_encoder_out': vis_encoder_out,  # B, T, D embed_dim
            'visual_prelogits': visual_prelogits,
        }

        if self.args.image_verbose:
            print('ENCODER OUT: encoder_out %s' %
                  (str(encoder_out_dict['encoder_out'].shape)))

        if not self.args.image_disable:
            if self.args.image_verbose:
                print('ENCODER OUT: visual_prelogits %s' %
                      (str(visual_prelogits.shape)))

        return encoder_out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out.get('encoder_states', None) is not None:
            for idx, state in enumerate(encoder_out['encoder_states']):
                encoder_out['encoder_states'][idx] = state.index_select(
                    1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(
                    self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


@register_model_architecture('visualtransformer', 'visualtransformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(
        args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(
        args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(
        args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(
        args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(
        args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('visualtransformer', 'visual_transformer_iwslt_de_en')
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
