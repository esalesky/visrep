import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ResNet(torch.nn.Module):
    """Define a ResNet architecture for face representation."""

    #default_dim = 128
    #default_input_shape = ImageShape(channels=3, height=112, width=112)

    def __init__(self, dim=None, nbr_classes=None,  # input_shape=None,
                 model_name='resnet18',
                 extract='layer4', log_softmax=True):
        super().__init__()
        self.dim = dim
        #self.input_shape = input_shape
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
