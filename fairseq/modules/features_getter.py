import torch.nn as nn
import fairseq.modules as modules

import logging

LOG = logging.getLogger(__name__)


class ConvFeaturesGetter(nn.Module):
    """CNN features getter for the Encoder of image."""

    def __init__(self, backbone_name, pretrained=False):
        super().__init__()
        # loading network
        LOG.info("loading network %s", backbone_name)

        conv_model_in = getattr(modules, backbone_name)(pretrained=pretrained)

        if backbone_name.startswith("resnet") or backbone_name.startswith("mobilenet"):
            conv = list(conv_model_in.children())[:-2]
            conv = nn.Sequential(*conv)
        elif backbone_name.startswith("densenet"):
            conv = list(conv_model_in.features.children())
            conv.append(nn.ReLU(inplace=True))
            conv = nn.Sequential(*conv)
        elif backbone_name.startswith("vista"):
            conv = conv_model_in
        else:
            raise ValueError(
                "Unsupported or unknown architecture: {}!".format(
                    backbone_name)
            )

        self.conv = conv

    def forward(self, x):
        return self.conv(x)
