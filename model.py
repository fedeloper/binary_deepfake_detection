#!/usr/bin/env python3
import torch
#!/usr/bin/env python3
from collections import OrderedDict
import cv2 as cv
import numpy as np 
import einops
from skimage import feature

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, auroc
import timm
import lightning as L
from BNext.src.bnext import BNext

# from backbones.caddm import CADDM



def get(labels, pretrained_model=None, backbone='BNext-T', freeze_backbone=True, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=True, pos_weight=True):
    if backbone not in ['BNext-L', 'BNext-T', 'BNext-S','BNext-M']:
        raise ValueError("Unsupported type of models!")

    model = CADDM(num_classes=labels, backbone=backbone, freeze_backbone=freeze_backbone, 
                  add_magnitude_channel=add_magnitude_channel, add_fft_channel=add_fft_channel, add_lbp_channel=add_lbp_channel)

    # if pretrained_model:
    #     checkpoint = torch.load(pretrained_model)
    #     model.load_state_dict(checkpoint['network'])
    return model

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

class CADDM(L.LightningModule):

    def __init__(self, num_classes, backbone='BNext-T', 
                 freeze_backbone=True, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=True,
                 learning_rate=1e-4, pos_weight=1.):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
        
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
        
        # update the preprocessing metas
        assert isinstance(add_magnitude_channel, bool)
        self.add_magnitude_channel = add_magnitude_channel
        assert isinstance(add_fft_channel, bool)
        self.add_fft_channel = add_fft_channel
        assert isinstance(add_lbp_channel, bool)
        self.add_lbp_channel = add_lbp_channel
        self.new_channels = sum([self.add_magnitude_channel, self.add_fft_channel, self.add_lbp_channel])
        
        # loss parameters
        self.pos_weight = pos_weight
        
        if self.new_channels > 0:
            self.adapter = nn.Conv2d(in_channels=3+self.new_channels, out_channels=3, 
                                     kernel_size=3, stride=1, padding=1)
        else:
            self.adapter = nn.Identity()
            
        # disables the last layer of the backbone
        self.inplanes = self.base_model.fc.in_features
        self.base_model.deactive_last_layer=True
        self.base_model.fc = nn.Identity()

        # eventually freeze the backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # add a new linear layer after the backbone
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        outs = {}
        # eventually concat the edge sharpness to the input image in the channel dimension
        if self.add_magnitude_channel or self.add_fft_channel or self.add_lbp_channel:
            x = self.add_new_channels(x)

        # extracts the features
        x_adapted = self.adapter(x)
        
        # normalizes the input image
        x_adapted = (x_adapted - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)
        features = self.base_model(x_adapted)
        
        # outputs the logits
        outs["logits"] = self.fc(features)
        return outs
    
    def configure_optimizers(self):
        modules_to_train = [self.adapter, self.fc]
        if not self.freeze_backbone:
            modules_to_train.append(self.base_model)
        optimizer = optim.AdamW(
            [parameter for module in modules_to_train for parameter in module.parameters()], 
            lr=self.learning_rate,
            )
        return optimizer
    
    def _add_new_channels_worker(self, image):
        # convert the image to grayscale
        gray = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
        
        new_channels = []
        if self.add_magnitude_channel:
            # calculate x and y gradients using sobel operator
            sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=7)
            sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=7)
            
            # calculate magnitude of the gradients
            magnitude = np.sqrt(sobelx**2 + sobely**2) 
            new_channels.append(magnitude)
        
        #if fast_fourier is required, calculate it
        if self.add_fft_channel:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            fft_spectrum = 20*np.log(np.abs(fshift) + 1e-9)
            new_channels.append(fft_spectrum)
        
        #if localbinary pattern is required, calculate it
        if self.add_lbp_channel:
            lbp = feature.local_binary_pattern(gray, 3, 6, method='uniform')
            new_channels.append(lbp)

        new_channels = np.stack(new_channels, axis=2) / 255
        return new_channels
        
    def add_new_channels(self, images):
        device = images.device
        #copy the input image to avoid modifying the originalu
        images_copied = einops.rearrange(images.clone().cpu().numpy(), "b c h w -> b h w c")
        
        # parallelize over each image in the batch using pool
        new_channels = np.stack([self._add_new_channels_worker(image) for image in images_copied], axis=0)
        
        # concatenates the new channels to the input image in the channel dimension
        images_copied = np.concatenate([images_copied, new_channels], axis=-1)
        # cast img again to torch tensor and then reshape to (B, C, H, W)
        images_copied = einops.rearrange(torch.from_numpy(images_copied).float().to(device), "b h w c -> b c h w")
        return images_copied
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def training_step(self, batch, i_batch):
        return self.step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self.step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self.step(batch, i_batch, phase="test")
    
    def on_train_epoch_end(self):
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
    
    def step(self, batch, i_batch, phase=None):
        images = batch["image"].to(self.device)
        outs = {
            "phase": phase,
            "labels": batch["is_real"][:, 0].float().to(self.device),
        }
        outs.update(self(images))
        if self.num_classes == 2:
            outs["loss"] = F.binary_cross_entropy_with_logits(input=outs["logits"][:, 0], target=outs["labels"], pos_weight=torch.as_tensor(self.pos_weight, device=self.device))
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        # transfer each tensor to cpu previous to saving them
        for k in outs:
            if (not "loss" in k) and isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()
        # saves the outputs
        self.epoch_outs.append(outs)
        return outs
    
    def _on_epoch_start(self):
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        with torch.no_grad():
            labels = torch.cat([batch["labels"] for batch in self.epoch_outs], dim=0)
            logits = torch.cat([batch["logits"] for batch in self.epoch_outs], dim=0)[:, 0]
            phases = [phase for batch in self.epoch_outs for phase in [batch["phase"]] * len(batch["labels"])]
            assert len(labels) == len(logits), f"{len(labels)} != {len(logits)}"
            assert len(phases) == len(labels), f"{len(phases)} != {len(labels)}"
            for phase in ["train", "val", "test"]:
                indices_phase = [i for i in range(len(phases)) if phases[i] == phase]
                if len(indices_phase) == 0:
                    continue                
                metrics = {
                    "acc": accuracy(preds=logits[indices_phase], target=labels[indices_phase], task="binary", average="micro"),
                    "auc": auroc(preds=logits[indices_phase], target=labels[indices_phase].long(), task="binary", average="micro"),
                }
                for metric in metrics:
                    self.log(name=f"{phase}_{metric}", value=metrics[metric], prog_bar=True, logger=True)
         
        
if __name__ == "__main__":
    model = get(labels=2)
    # runs a dummy forward pass to check if the model is working properly
    model(torch.randn(8, 3, 224, 224))
