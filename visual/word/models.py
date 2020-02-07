import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import logging

LOG = logging.getLogger(__name__)


class VisualNet(torch.nn.Module):
    """Define an architecture for visual word representation. """

    OUTPUT_SCALE = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
    CHANNEL_SCALE = {'resnet18': 512, 'resnet50': 2048}

    def __init__(self, dim=None, input_shape=None, model_name='resnet18',
                 extract='layer4', normalize=False):
        super().__init__()
        self.dim = dim
        self.model_name = model_name
        self.model_class = getattr(torchvision.models, model_name)
        self.model = self.model_class(num_classes=dim)
        self.extract = extract

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

        LOG.info('initialized ResNet')

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
        x = self.top(x)
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
        LOG.info('initialized TopLayer4')

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
        LOG.info('initialized TopAvgPool')

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
        LOG.info('initialized TopFC')

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
        LOG.info('initialized Softmax')

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
