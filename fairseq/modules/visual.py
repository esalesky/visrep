import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class DirectOCR(nn.Module):
    """
    Directly encodes pixels into the embedding
    size via a linear transformation.
    """
    def __init__(self,
                 slice_width,
                 slice_height,
                 embed_dim):

        super().__init__()
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.embed_dim = embed_dim

        self.embedder = nn.Linear(slice_width * slice_height,
                                  embed_dim)
        logger.info(f"embedding from {slice_width} * {slice_height} = {slice_width * slice_height} to {embed_dim}")

    def forward(self, image_slice):
        """
        An image_slice is a piece of a rendered sentence with a fixed
        width and height. Here we directly map to the embedding size with
        a big ol' linear layer.

        The slice width comes from --image-window. The height is computed
        from the font size (usually about 21 for an 8 point font).
        """
        # Assume there is just one channel so it can be removed
        batch_size, src_len, channels, width, height = image_slice.shape
        pixels = image_slice.view(batch_size * src_len, width * height)

        # Embed and recast to 3d tensor
        embeddings = self.embedder(pixels)
        embeddings = embeddings.view(batch_size, src_len, self.embed_dim)

        return embeddings


class VistaOCR(nn.Module):
    """ Vista OCR - VGG style """

    def __init__(self, use_bridge, encoder_dim, input_line_height, image_type, kernel_size=2, width=0.7, use_pool=False, image_verbose=True):
        super().__init__()

        self.use_bridge = use_bridge
        self.encoder_dim = encoder_dim
        self.input_line_height = input_line_height
        self.image_verbose = image_verbose
        self.image_type = image_type
        self.use_pool = use_pool
        self.kernel_size = kernel_size
        self.width = width

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(3, 64),
            *self.ConvBNReLU(64, 64),
            nn.FractionalMaxPool2d(self.kernel_size, output_ratio=(0.5, self.width)),
            *self.ConvBNReLU(64, 128),
            *self.ConvBNReLU(128, 128),
            nn.FractionalMaxPool2d(self.kernel_size, output_ratio=(0.5, self.width)),
            *self.ConvBNReLU(128, 256),
            *self.ConvBNReLU(256, 256),
            *self.ConvBNReLU(256, 256)
        )

        if self.use_bridge:
            print('VistaOCR: use bridge')

            # We need to calculate cnn output size to construct the bridge layer
            fake_input_width = 800
            print('VistaOCR: Fake input width %d' % (fake_input_width))
            cnn_out_h, cnn_out_w = self.cnn_input_size_to_output_size(
                (self.input_line_height, fake_input_width))
            print('VistaOCR: CNN out height %d, width %d' %
                  (cnn_out_h, cnn_out_w))
            cnn_out_c = self.cnn_output_num_channels()

            cnn_feat_size = cnn_out_c * cnn_out_h

            print('VistaOCR: CNN out height %d' % (cnn_out_h))
            print('VistaOCR: CNN out channels %d' % (cnn_out_c))
            print('VistaOCR: CNN feature size (channels %d x height %d) = %d' %
                  (cnn_out_c, cnn_out_h, cnn_feat_size))

            self.bridge_layer = nn.Sequential(
                nn.Linear(cnn_feat_size, self.encoder_dim),
                nn.ReLU(inplace=True)
            )

            if self.use_pool:
                print("Applying pooling to reduce dimensions")
                self.avgpool = nn.AdaptiveAvgPool2d((None, 1))

        else:
            print('VistaOCR: avg pool')
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        With word-based decoding, the data is a batch of images with the following shape: (batch x
        tokens, channel, height, width).  The output is (batch x tokens, d=embed_dim).

        For line-based decoding: Input shape = (batch, channel, height, width)
        Output shape = (batch x slices, d=embed_dim)

        """
        if self.image_verbose:
            print('VistaOCR: input', x.shape)

        x = self.cnn(x)

        if self.image_verbose:
            print('VistaOCR: forward features out', x.shape)

        if self.use_bridge:
            b, c, h, w = x.size()
            x = x.permute(3, 0, 1, 2).contiguous()
            x = x.view(-1, c * h)
            x = self.bridge_layer(x)
            if self.image_verbose:
                print('VistaOCR: forward bridge out', x.shape)

            if self.image_type == "word":
                # (width * batch * word, model_size) -> (width, batch * word, model_size)
                x = x.view(w, b, -1)
            elif self.image_type == "line":

                x = x.view(b, w, -1)

            if self.image_verbose:
                print('VistaOCR: forward bridge view', x.shape)

            if self.use_pool:
                # (width, batch * word, model_size) -> (batch * word, model_size, width)
                x = x.permute(1, 2, 0).contiguous()
                if self.image_verbose:
                    print('VistaOCR: forward permute out', x.shape)
                # (batch * word, model_size, width) -> (batch * word, model_size, 1)
                x = self.avgpool(x).squeeze()
                if self.image_verbose:
                    print('VistaOCR: forward avg pool out', x.shape)
        else:
            x = self.avgpool(x)
            if self.image_verbose:
                print('VistaOCR: forward avg pool out', x.shape)
            x = x.squeeze()
            if self.image_verbose:
                print('VistaOCR: squeeze out', x.shape)

        return x

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c

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

            out_h = math.floor(
                (out_h + 2.0 * padding_y - dilation_y *
                 (kernel_size_y - 1) - 1) / stride_y + 1)
            out_w = math.floor(
                (out_w + 2.0 * padding_x - dilation_x *
                 (kernel_size_x - 1) - 1) / stride_x + 1)
        elif isinstance(module, nn.FractionalMaxPool2d):
            if module.output_size is not None:
                out_h, out_w = module.output_size
            else:
                rh, rw = module.output_ratio
                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)

        return out_h, out_w

    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size

        for module in self.cnn.modules():
            out_h, out_w = self.calculate_hw(module, out_h, out_w)

        return (out_h, out_w)


class VisualNet(torch.nn.Module):
    """Define an architecture for visual word representation. """

    OUTPUT_SCALE = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
    CHANNEL_SCALE = {'resnet18': 512, 'resnet50': 2048}

    def __init__(self, dim=None, input_shape=None, model_name='resnet18',
                 extract='layer4', normalize=False, image_verbose=True):
        super().__init__()
        self.dim = dim
        self.model_name = model_name
        self.model_class = getattr(torchvision.models, model_name)
        try:
            self.model = self.model_class(num_classes=dim)
        except TypeError as e:
            import sys
            print(f"No such model type '{model_name}': bad argument to --image-backbone", file=sys.stderr)
            sys.exit(1)

        self.extract = extract
        self.image_verbose = image_verbose

        if extract == 'layer4':
            del self.model.fc
            channels = self.CHANNEL_SCALE[model_name]
            height = input_shape[0] / self.OUTPUT_SCALE['layer4']
            width = input_shape[1] / self.OUTPUT_SCALE['layer4']
            out_shape = (None, channels, math.ceil(height), math.ceil(width))
            self.top = TopLayer4(out_shape, dim=dim, normalize=normalize)
        elif extract == 'avgpool':
            del self.model.fc
            out_shape = (None, 512)
            self.top = TopAvgPool(out_shape, dim=dim, normalize=normalize)
        elif extract == 'fc':
            out_shape = (None, dim)
            self.top = TopFC(out_shape, dim=dim, normalize=normalize)
        else:
            raise ValueError('unknown ResNet layer name given to extract')

        print('initialized ResNet')

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if self.image_verbose:
            print('ENCODER: VisualNet after conv/bn/relu ', x.shape)
        x = self.model.maxpool(x)
        if self.image_verbose:
            print('ENCODER: VisualNet after maxpool ', x.shape)

        done = self.extract == 'maxpool'
        if not done:
            x = self.model.layer1(x)
            done = self.extract == 'layer1'
            if self.image_verbose:
                print('ENCODER: VisualNet after layer1 ', x.shape)
        if not done:
            x = self.model.layer2(x)
            done = self.extract == 'layer2'
            if self.image_verbose:
                print('ENCODER: VisualNet after layer2 ', x.shape)
        if not done:
            x = self.model.layer3(x)
            done = self.extract == 'layer3'
            if self.image_verbose:
                print('ENCODER: VisualNet after layer3 ', x.shape)
        if not done:
            x = self.model.layer4(x)
            done = self.extract == 'layer4'
            if self.image_verbose:
                print('ENCODER: VisualNet after layer4 ', x.shape)
        if not done:
            x = self.model.avgpool(x)
            x = x.reshape(x.size(0), -1)
            done = self.extract == 'avgpool'
            if self.image_verbose:
                print('ENCODER: VisualNet after avgpool ', x.shape)
        if not done:
            x = self.model.fc(x)
        x = self.top(x)
        if self.image_verbose:
            print('ENCODER: VisualNet return ', x.shape)
        return x


class TopLayer4(torch.nn.Module):
    """Define a subset of a network to finalize features from a backbone."""

    def __init__(self, input_shape, dim=512, dropout_prob=0.4, normalize=False):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
        self.num_activations = input_shape[1] * input_shape[2] * input_shape[3]

        self.bn1 = torch.nn.BatchNorm2d(input_shape[1])
        self.dropout = torch.nn.Dropout2d(p=dropout_prob)
        self.linear = torch.nn.Linear(self.num_activations, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        print('initialized TopLayer4')

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = self.bn2(x)
        if self.normalize:
            x = F.normalize(x)
        return x


class TopAvgPool(torch.nn.Module):
    """Define a subset of a network to finalize features from a backbone."""

    def __init__(self, input_shape, dim=512, dropout_prob=0.4, normalize=False):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
        self.num_activations = input_shape[1]

        self.linear = torch.nn.Linear(self.num_activations, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        print('initialized TopAvgPool')

    def forward(self, x):
        x = self.linear(x)
        x = self.bn2(x)
        if self.normalize:
            x = F.normalize(x)
        return x


class TopFC(torch.nn.Module):
    """Define a subset of a network to finalize features from a backbone."""

    def __init__(self, input_shape, dim=512, dropout_prob=0.4, normalize=False):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
        self.num_activations = input_shape[1]

        self.linear = torch.nn.Linear(self.num_activations, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        print('initialized TopFC')

    def forward(self, x):
        x = self.linear(x)
        x = self.bn2(x)
        if self.normalize:
            x = F.normalize(x)
        return x


class Softmax(nn.Module):
    """Define the final layer for training a multi-class classifier."""

    def __init__(self, dim, dim_out, log_softmax=False):
        """Create an instance of the Softmax head."""
        super().__init__()
        self.log_softmax = log_softmax
        self.linear = nn.Linear(dim, dim_out)
        print('initialized Softmax')

    def forward(self, x):  # pylint: disable=unused-argument
        """Compute logits from representations for training."""
        x = self.linear(x)

        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return x


class VisualTrainer(nn.Module):
    """Combine a backbone with a head for training."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        backbone = self.backbone(x)
        return backbone, self.head(backbone)
