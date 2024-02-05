#!/usr/bin/env python3
from collections import OrderedDict
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from backbones.adm import Artifact_Detection_Module

from backbones.efficientnet_pytorch import EfficientNet
from BNext.src.bnext import BNext
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict
class CADDM(nn.Module):

    def __init__(self, num_classes, backbone='BNext-T'):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            size = size_map[backbone]
            self.base_model = BNext(num_classes=1000, size=size)
            checkpoint = torch.load(f"pretrained/{size}_checkpoint.pth.tar", map_location="cpu")
            self.base_model.load_state_dict(remove_data_parallel(checkpoint))
        else:
            print(backbone)
            raise ValueError("Unsupported Backbone!")
        self.base_model.deactive_last_layer=True
        self.inplanes = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        # self.inplanes = self.base_model.out_num_features

        # self.adm = Artifact_Detection_Module(self.inplanes)

        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        features = self.base_model(x)
        logits = self.fc(features)

        if self.num_classes >= 3:
            if self.training:
                return logits
            return torch.softmax(logits, dim=-1)
        else:
            logits = logits[:, 0]
            if self.training:
                return logits
            return torch.sigmoid(logits)
