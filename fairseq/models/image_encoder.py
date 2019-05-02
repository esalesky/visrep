
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import math
import logging

logger = logging.getLogger('root')


'''

Image encoder

CNN Input
 [1, 3, 30, 135]  # batch, channels, height, width
CNN Output / Bridge input
 [1, 256, 7, 65]  # batch, channels, height, width
Bridge output
 [65, 1, 512]     # width, batch, features
 
'''

class ImageEncoder(nn.Module):
    
    def __init__(self, input_channels=3, output_dim=512):
        super(ImageEncoder, self).__init__()
        
        self.bridge_output_dim = output_dim # 512
        self.num_in_channels = input_channels # 3
        
        self.cnn = nn.Sequential(
            *self.ConvBNReLU(self.num_in_channels , 64),
            *self.ConvBNReLU(64, 64),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(64, 128),
            *self.ConvBNReLU(128, 128),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(128, 256),
            *self.ConvBNReLU(256, 256),
            *self.ConvBNReLU(256, 256)
        )
        
        fake_input_width = 600
        cnn_out_h, cnn_out_w = self.cnn_input_size_to_output_size((self.input_line_height, fake_input_width))
        cnn_out_c = self.cnn_output_num_channels()
        cnn_feat_size = cnn_out_c * cnn_out_h
        
        logger.info('CNN out height %d' % (cnn_out_h))
        logger.info('CNN out channels %d' % (cnn_out_c))
        logger.info('CNN feature size (channels %d x height %d) = %d' % (cnn_out_c, cnn_out_h, cnn_feat_size))
                
        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.bridge_output_dim),
            nn.ReLU(inplace=True)
        )

        # initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

        total_params = 0
        for param in self.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            total_params += local_params

        cnn_params = 0
        for param in self.cnn.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            cnn_params += local_params
            
        logger.info("Total Model Params = %d" % total_params)
        logger.info("\tCNN Params = %d" % cnn_params)

        logger.info("Model looks like:")
        logger.info(repr(self))

        if torch.cuda.device_count() > 0:   
            logger.info('...available %d GPUs' % torch.cuda.device_count())  
        else:
            logger.info('...available CPU')
            
        if torch.cuda.is_available() and self.gpu:
            logger.info('...using GPUS')
            self.cnn = self.cnn.cuda()
            self.bridge_layer = self.bridge_layer.cuda()

            if self.multigpu:
                print('...using MULTIPLE GPUS...torch.nn.DataParallel')
                self.cnn = torch.nn.DataParallel(self.cnn)
        else:
            logger.info.info("Warning: Running model on CPU")

                        
    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]


    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size

        for module in self.cnn.modules():
            out_h, out_w = self.calculate_hw(module, out_h, out_w)

        return (out_h, out_w)
    
    
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

            out_h = math.floor((out_h + 2.0 * padding_y - dilation_y * (kernel_size_y - 1) - 1) / stride_y + 1)
            out_w = math.floor((out_w + 2.0 * padding_x - dilation_x * (kernel_size_x - 1) - 1) / stride_x + 1)
        elif isinstance(module, nn.FractionalMaxPool2d):
            if module.output_size is not None:
                out_h, out_w = module.output_size
            else:
                rh, rw = module.output_ratio
                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)

        return out_h, out_w


    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c

        
    def forward(self, x):
        cnn_output = self.cnn(x)

        b, c, h, w = cnn_output.size()
        #print('forward -> cnn size b %d, c %d, h %d, w %d' % (b, c, h, w))
        
        cnn_output = cnn_output.permute(3, 0, 1, 2).contiguous()

        features = self.bridge_layer(cnn_output.view(-1, c * h)).view(w, b, -1)
        #print('forward -> bridge size ', features.size())
        
        return features
    
    
            
    