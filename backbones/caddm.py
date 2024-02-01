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
        self.inplanes = self.base_model.out_num_features

        self.adm = Artifact_Detection_Module(self.inplanes)

        self.fc = nn.Linear(self.inplanes, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_num = x.size(0)
        x, global_feat = self.base_model(x)

        # location result, confidence of each anchor, final feature map of adm.
        loc, cof, adm_final_feat = self.adm(x)

        final_cls_feat = global_feat + adm_final_feat
        final_cls = self.fc(final_cls_feat.view(batch_num, -1))

        if self.training:
            return loc, cof, final_cls
        return self.softmax(final_cls)

# vim: ts=4 sw=4 sts=4 expandtab
