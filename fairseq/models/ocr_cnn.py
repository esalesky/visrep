
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

#from fairseq import options
#from ocr.models.ocr_encoder import OCREncoder
#from fairseq.models.lstm import LSTM
from fairseq.models.lstm import LSTMDecoder
#from collections import OrderedDict

#import ocr.modules as modules
from fairseq.modules.densenet import densenet121
from fairseq.modules.resnet import resnet50
from fairseq.modules.seresnext import se_resnet50
from fairseq.modules.vgg import vgg_vista
import numpy as np
import torch.autograd as autograd



#FEATURES = {}


@register_model('ocr_cnn')
class OCRCNNModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--encoder-arch', default='seresnext',
                            help='CNN architecture. (default: seresnext)')
        #parser.add_argument('--decoder-arch', default='ocrcnndecoder',
        #                    help='Decoder architecture. (lstmdecoder or ocrcnnecoder)')
        
        parser.add_argument("--encoder-dim", type=int, default=128, 
                            help="Encoder output dimension")
        
        parser.add_argument("--lstm-layers", type=int, default=2, 
                            help="Number of LSTM layers in model")
#         parser.add_argument("--lstm-hidden-size", type=int, default=640, 
#                             help="The number of features in the hidden state `h`")
#         parser.add_argument("--lstm-dropout", type=float, default=0.50, 
#                             help="The number of features in the hidden state `h`")
#         parser.add_argument('--lstm-bidirectional', action='store_true',
#                             default=True,
#                             help='make all layers of decoder bidirectional')

        parser.add_argument('--attention', action='store_true', default=False, 
                            help='Use attention')    

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        #print('criterion {}'.format(args.criterion))
        encoder = OCRCNNEncoder(
            args=args,
        )
        
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.encoder_dim,
            hidden_size= args.encoder_dim, # has to be this or breaks
            num_layers=args.lstm_layers,
            encoder_output_units=args.encoder_dim,
            attention=args.attention,
        )
        return cls(encoder, decoder)


    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)        
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
        return decoder_out


class OCRCNNEncoder(FairseqEncoder):
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
                
        #cnn_out_h, _ = self.cnn_input_size_to_output_size((self.input_line_height, self.fake_input_width))
        #cnn_out_c = self.cnn_output_num_channels()

        #cnn_feat_size = cnn_out_c * cnn_out_h

#         ''' calculate cnn output channels and height for bridge layer '''
#         test_x = torch.Tensor(1,3, self.input_line_height, self.fake_input_width) # shape = (batch size, channels, height, width)
#         test_run = self.cnn(autograd.Variable(torch.Tensor(test_x)))
#         cnn_out_c = int(np.prod(test_run.size()[1]))
#         cnn_out_h = int(np.prod(test_run.size()[2]))
#         cnn_feat_size = cnn_out_c * cnn_out_h
#         print(test_run.size(), cnn_out_c, cnn_out_h, cnn_feat_size)
        #print(compute_out_size(test_x.size(), self))

        print('CNN out height %d' % (cnn_out_h))
        print('CNN out channels %d' % (cnn_out_c))
        print('CNN feature size %d' % (cnn_feat_size))

        #self.embed_dim = self.lstm_input_dim #cnn_out_c
        #print('self.embed_dim', self.embed_dim )

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.embed_dim),
            nn.ReLU(inplace=True)
        )

        hidden_size = args.encoder_dim
        
        init_layers = args.lstm_layers
        if args.lstm_bidirectional:
            init_layers *= 2
        
        self.init_hidden_w = nn.Parameter(
            torch.rand(init_layers, self.embed_dim, hidden_size)
        )  # init_layers x embed_dim x hidden_size
        self.init_hidden_b = nn.Parameter(
            torch.rand(init_layers, 1, hidden_size)
        )  # init_layers x 1 x hidden_size
        self.init_cell_w = nn.Parameter(torch.rand_like(self.init_hidden_w))
        self.init_cell_b = nn.Parameter(torch.rand_like(self.init_hidden_b))
                    
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

        final_hiddens, final_cells = self.init_hidden(x)
        
        return {
            'encoder_out': (x, final_hiddens, final_cells),
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
        
                

@register_model_architecture('ocr_cnn', 'ocr_cnn')
def base_architecture(args):
    args.encoder_arch = getattr(args, 'encoder_arch', 'seresnext')
    #args.decoder_arch = getattr(args, 'decoder_arch', 'ocrcnnecoder')
    args.encoder_dim = getattr(args, 'encoder_dim', 128)
    args.lstm_layers = getattr(args, 'lstm_layers', 2)
    args.lstm_hidden_size = getattr(args, 'lstm_hidden_size', 640)
    args.lstm_dropout = getattr(args, 'lstm_dropout', 0.50)
    args.lstm_bidirectional = getattr(args, 'lstm_bidirectional', True)
    args.attention = getattr(args, 'attention', True)

