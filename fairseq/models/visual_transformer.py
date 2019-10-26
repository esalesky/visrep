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
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class FairseqVisualEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        # assert isinstance(self.encoder, FairseqEncoder)
        # assert isinstance(self.decoder, FairseqDecoder)

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
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs)
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


@register_model('visualtransformer')
class VisualTransformerModel(FairseqVisualEncoderDecoderModel):
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

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
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

        print(args)
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        # return TransformerEncoder(args, src_dict, embed_tokens)
        return VisualTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class ResNet(torch.nn.Module):
    """Define a ResNet architecture for face representation."""

    def __init__(self, dim=None, nbr_classes=None,  # input_shape=None,
                 model_name='resnet18',
                 extract='layer4', log_softmax=True):
        super().__init__()
        self.dim = dim
        # self.input_shape = input_shape
        self.model_name = model_name
        self.log_softmax = log_softmax
        self.model_class = getattr(torchvision.models, model_name)
        self.model = self.model_class(num_classes=dim)
        self.extract = extract
        if extract == 'layer4':
            del self.model.fc
            out_shape = (None, 512, 4, 4)
        elif extract == 'avgpool':
            del self.model.fc
            out_shape = (None, 512)
            dim = 512
        elif extract == 'fc':
            out_shape = (None, dim)
        else:
            raise ValueError('unknown ResNet layer name given to extract')
        if log_softmax:
            self.linear = nn.Linear(dim, nbr_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        done = self.extract == 'maxpool'
        if not done:
            x = self.model.layer1(x)
            done = self.extract == 'layer1'
        if not done:
            x = self.model.layer2(x)
            done = self.extract == 'layer2'
        if not done:
            x = self.model.layer3(x)
            done = self.extract == 'layer3'
        if not done:
            x = self.model.layer4(x)
            done = self.extract == 'layer4'
        if not done:
            x = self.model.avgpool(x)
            x = x.reshape(x.size(0), -1)
            done = self.extract == 'avgpool'
        if not done:
            x = self.model.fc(x)
        if self.log_softmax:
            x = self.linear(x)
            x = F.log_softmax(x, dim=-1)
        return x


class VisualFairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths=None, src_images=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class VisualTransformerEncoder(VisualFairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.image_sample_write = False
        self.image_sample_write_ctr = 0
        self.image_samples_path = args.image_samples_path

        if self.image_samples_path:
            self.image_sample_write = True
            self.image_samples_path = self.image_samples_path + '/samples_train'
            if not os.path.exists(self.image_samples_path):
                os.makedirs(self.image_samples_path)

        self.robust_encoder = ResNet(
            dim=512, extract='avgpool', log_softmax=False)

        self.dropout = args.dropout

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

    def visual_forward_embedding(self, src_tokens, src_images):
        # embed tokens and positions
        b, t, c, h, w = src_images.shape  # batch x token x channel x height x width
        # image shape torch.Size([40, 73, 3, 32, 128])  # B X T X C X H X W
        #print('src_images', src_images.shape)
        src_images = src_images.view(-1, src_images.size(-3),
                                     src_images.size(-2), src_images.size(-1))
        #print('src_images reshape', src_images.shape)
        embed = self.robust_encoder(src_images)
        #print('robust ', embed.shape)
        embed = embed.view(b, t, 512)
        #print('robust reshape', embed.shape)

        embed = self.embed_scale * embed
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def visual_print_samples(self, src_tokens, src_images):
        if self.image_sample_write:
            for ctr in range(len(src_images[0])):
                token_id = src_tokens[0][ctr].cpu()
                word = self.dictionary[int(token_id)]

                image = np.uint8(src_images[0][ctr].cpu(
                ).squeeze()).transpose((1, 2, 0)) * 255

                outimage = self.image_samples_path + '/' + \
                    str(int(token_id)) + '_' + str(word) + '.png'
                cv2.imwrite(outimage, image)

            self.image_sample_write_ctr += 1
            if self.image_sample_write_ctr > 5:
                self.image_sample_write = False

    def forward(self, src_tokens, src_lengths, src_images=None, cls_input=None, return_all_hiddens=False):
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

        self.visual_print_samples(src_tokens, src_images)

        x, encoder_embedding = self.visual_forward_embedding(
            src_tokens, src_images)
        # print('x shape', x.shape)

        # x, encoder_embedding = self.forward_embedding(src_tokens)
        # print('x shape', x.shape)

        # Example shapes from encoder:
        # src_images torch.Size([240, 12, 3, 32, 128])  # B X T X C X H X W
        # src_images reshape torch.Size([2880, 3, 32, 128])  # (B X T) X C X H X W
        # robust  torch.Size([2880, 512])  # (B X T) X embed_dim
        # robust reshape torch.Size([240, 12, 512])  # B X T X embed_dim  visual
        # image_x shape torch.Size([240, 12, 512]) # B X T X embed_dim  visual_forward_embedding
        # x shape torch.Size([240, 12, 512])  # B X T X embed_dim  forward_embedding

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

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
        }

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
