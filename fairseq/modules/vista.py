import torch
import torch.nn as nn


class VistaOCR(nn.Module):
    """ Vista OCR - VGG style """

    def __init__(self):
        super().__init__()

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

    def forward(self, x):
        x = self.cnn(x)
        return x

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [
            nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
            nn.BatchNorm2d(nOutputMaps),
            nn.ReLU(inplace=True),
        ]


def vista(pretrained=False, progress=True, **kwargs):
    model = VistaOCR(**kwargs)
    return model
