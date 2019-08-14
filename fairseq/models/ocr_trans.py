
import math

import torch

import torch.nn as nn

from fairseq import utils
from . import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder, Embedding
#from fairseq import options
#from ocr.models.ocr_encoder import OCREncoder
#from fairseq.models.lstm import LSTM
#from fairseq.models.lstm import LSTMDecoder
#from collections import OrderedDict

#import ocr.modules as modules
from fairseq.modules.densenet import densenet121
from fairseq.modules.resnet import resnet50
from fairseq.modules.seresnext import se_resnet50
from fairseq.modules.vgg import vgg_vista
import numpy as np
import torch.autograd as autograd



#FEATURES = {}

DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('ocr_trans')
class OCRTransModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--encoder-arch', default='seresnext',
                            help='CNN architecture. (default: seresnext)')
        parser.add_argument("--encoder-dim", type=int, default=128, 
                            help="Encoder output dimension")
        
        #parser.add_argument('--activation-fn',
        #                    choices=utils.get_available_activation_fns(),
        #                    help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        #parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
        #                    help='dropout probability after activation in FFN.')
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
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-token-rnn', default=False, action='store_true',
                            help='if set, disables rnn layer')
        parser.add_argument('--no-token-crf', default=False, action='store_true',
                            help='if set, disables conditional random fields')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        #print('criterion {}'.format(args.criterion))

        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
                    
        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.encoder_dim, None,
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embed_tokens)
        return OCRTransModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return OCRTransEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)
    

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)        
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
        return decoder_out


class OCRTransEncoder(FairseqEncoder):
    def __init__(self, args):
        super(FairseqEncoder, self).__init__()
        
        self.args = args
        
        self.embed_dim = args.encoder_dim 
        
        if self.args.encoder_arch.startswith('vista'):
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
        else:      
            self.cnn = nn.Sequential(*self.cnn_layers(args.encoder_arch))        
            
        #self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        
        self.fake_input_width = 59
        self.input_line_height = args.height
        self.lstm_input_dim = args.encoder_dim

        if self.args.encoder_arch.startswith('vista'):
            cnn_out_c = 256
            cnn_out_h = 8
        elif self.args.encoder_arch.startswith('densenet'):
            cnn_out_c = 1024
            cnn_out_h = 4
        elif self.args.encoder_arch.startswith('resnet'):
            cnn_out_c = 2048
            cnn_out_h = 4
        elif self.args.encoder_arch.startswith('seresnext'):
            cnn_out_c = 1024
            cnn_out_h = 8
        elif self.args.encoder_arch.startswith('vgg'):
            cnn_out_c = 256
            cnn_out_h = 8
        else:
            cnn_out_c = 256
            cnn_out_h = 8
            
        cnn_feat_size = cnn_out_c * cnn_out_h
                
        print('CNN out height %d' % (cnn_out_h))
        print('CNN out channels %d' % (cnn_out_c))
        print('CNN feature size %d' % (cnn_feat_size))

        #self.embed_dim = self.lstm_input_dim #cnn_out_c
        #print('self.embed_dim', self.embed_dim )

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.embed_dim),
            nn.ReLU(inplace=True)
        )

                    
    def forward(self, src_tokens):
        #print('\n src_tokens', src_tokens.size())     
        x = self.cnn(src_tokens)  # bsz x embed_dim x H' x W', where W' stands for `seq_len`
        #print('x features {}'.format(x.size()))

        ''' Remove avg pool, instead use bridge layer '''        
        #x = self.avgpool(x)
        #print('x avgpool {}'.format(x.size()))
        #x = x.permute(3, 0, 1, 2).view(x.size(3), x.size(0), -1)  # seq_len x bsz x embed_dim
        #print('x permute {}'.format(x.size()))

        ''' Use bridge layer '''
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        #print('x permute {}'.format(x.size()))
        x = x.view(-1, c * h)
        #print('x view {}'.format(x.size()))
        x = self.bridge_layer(x)
        #print('x bridge_layer {}'.format(x.size()))            
        x = x.view(w, b, -1)
        #print('x final {}'.format(x.size()))            

#         final_hiddens, final_cells = self.init_hidden(x)
#         
#         return {
#             'encoder_out': (x, final_hiddens, final_cells),
#             'encoder_padding_mask': None,
#         }

        return {
            'encoder_out': x,
            'encoder_padding_mask': None,
        }
        
        
    def init_hidden(self, x):
        mean = torch.mean(x, dim=0)  # bsz x embed_dim

        h0 = mean @ self.init_hidden_w + self.init_hidden_b  # init_layers x bsz x hidden_size
        h0 = torch.tanh(h0)

        c0 = mean @ self.init_cell_w + self.init_cell_b  # init_layers x bsz x hidden_size
        c0 = torch.tanh(c0)

        return (h0, c0)

    def max_positions(self):
        """Maximum sequence length supported by the encoder."""
        return 128


        
    @staticmethod
    def cnn_layers(cnn_arch):
        if cnn_arch.startswith('densenet'):
            net_in = densenet121()
            features = list(net_in.features.children())
            features.append(nn.ReLU(inplace=True))
        elif cnn_arch.startswith('resnet'):
            net_in = resnet50()
            features = list(net_in.children())[:-2]
        elif cnn_arch.startswith('seresnext'):          
            net_in = se_resnet50(layers=[3, 4, 6, 3], input_3x3=True, reduction=16, last_stride=1)
            features = list(net_in.children())[:-3]
        elif cnn_arch.startswith('vgg'):          
            net_in = vgg_vista()
            features = list(net_in.features.children())
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(cnn_arch)) 
                        
        return features
        
    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out
    
    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]
        
    def extract_features(self, src_tokens, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features
                    

@register_model_architecture('ocr_trans', 'ocr_trans')
def base_architecture(args):
    args.encoder_arch = getattr(args, 'encoder_arch', 'seresnext')
    #args.decoder_arch = getattr(args, 'decoder_arch', 'ocrcnnecoder')
    args.encoder_dim = getattr(args, 'encoder_dim', 128)
#    args.lstm_layers = getattr(args, 'lstm_layers', 2)
#    args.lstm_hidden_size = getattr(args, 'lstm_hidden_size', 640)
#    args.lstm_dropout = getattr(args, 'lstm_dropout', 0.50)
#    args.lstm_bidirectional = getattr(args, 'lstm_bidirectional', True)
#    args.attention = getattr(args, 'attention', True)

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    #args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    #args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    #args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.no_token_rnn = getattr(args, 'no_token_rnn', False)
    args.no_token_crf = getattr(args, 'no_token_crf', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.no_token_rnn = getattr(args, 'no_token_rnn', True)
    args.no_token_crf = getattr(args, 'no_token_crf', True)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    

