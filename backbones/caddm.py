#!/usr/bin/env python3
from collections import OrderedDict
import os
import cv2 as cv
import numpy as np 
import pywt
from skimage import feature

import torch
import torch.nn as nn
import torch.nn.functional as F
from BNext.src.bnext import BNext

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

class CADDM(nn.Module):

    def __init__(self, num_classes, backbone='BNext-T', freeze_backbone=True):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        
        # loads the backbone
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
        
        # disables the last layer of the backbone
        self.inplanes = self.base_model.fc.in_features
        self.base_model.deactive_last_layer=True
        self.base_model.fc = nn.Identity()

        # eventually freeze the backbone
        if freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # add a new linear layer after the backbone
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)


    def edge_sharpness(image:np.ndarray, retun_separate_gradients:bool=False, return_fast_fourier:bool=False, return_lbp:bool=False, join:bool=False):
        """
        This function calculates the edge sharpness of an image using the following methods:
        - Magnitude (Sobel X and Sobel Y)
        - Fast Fourier Transform
        - lbp (Local Binary Pattern)
        Parameters:
        - image: the input image (W, H, C)
        - retun_separate_gradients: if True, the function returns the magnitude, sobelx and sobely
        - return_fast_fourier: if True, the function returns the fast fourier transform of the image
        - return_lbp: if True, the function returns the local binary pattern of the image
        - join: if True, the function returns a single array with the magnitude, fast fourier and lbp
        Returns:
        - if join is True, the function returns a single array with the magnitude, fast fourier and lbp of shape (W, H, 3) 
        - if retun_separate_gradients is True, the function returns the magnitude, sobelx, sobely, fast fourier and lbp each of shape (W, H)
        - otherwise, the function returns the magnitude, fast fourier and lbp each of shape (W, H)
        """
        #copy the input image to avoid modifying the original
        img = image.copy()
        print(image.shape)
        #convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #calculate x and y gradients using sobel operator
        sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=7)
        sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=7)
        #calculate magnitude of the gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2) 

        #if fast_fourier is required, calculate it
        if return_fast_fourier:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            fft_spectrum = 20*np.log(np.abs(fshift))
        else:
            fft_spectrum = None
        
        #if localbinary pattern is required, calculate it
        if return_lbp:
            lbp = feature.local_binary_pattern(gray, 3, 6, method='uniform')
        else:
            lbp = None
        
        if join:
            return np.dstack((magnitude, fft_spectrum, lbp)), None, None
        elif retun_separate_gradients:
            return (magnitude, sobelx, sobely, fft_spectrum, lbp)
        else:
            return (magnitude, fft_spectrum, lbp)
    
    def forward(self, x):
        # extracts the features
        features = self.base_model(x)
        # outputs the logits
        logits = self.fc(features)

        # returns the logits in the appropriate manner
        if self.num_classes >= 3:
            # multiclass case
            if self.training:
                return logits
            return torch.softmax(logits, dim=-1)
        else:
            # binary case
            logits = logits[:, 0]
            if self.training:
                return logits
            return torch.sigmoid(logits)