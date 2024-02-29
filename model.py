import gc
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
from fvcore.nn import FlopCountAnalysis, parameter_count

from BNext.src.bnext import BNext

class BNext4DFR(L.LightningModule):

    def __init__(self, num_classes, backbone='BNext-T', 
                 freeze_backbone=True, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=True,
                 learning_rate=1e-4, pos_weight=1.):
        super(BNext4DFR, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
        
        # loads the backbone
        self.backbone = backbone
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            size = size_map[backbone]
            # loads the pretrained model
            self.base_model = nn.ModuleDict({"module": BNext(num_classes=1000, size=size)})
            pretrained_state_dict = torch.load(f"pretrained/{size}_checkpoint.pth.tar", map_location="cpu")
            self.base_model.load_state_dict(pretrained_state_dict)
            self.base_model = self.base_model.module
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
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=5)
        return [optimizer], [scheduler]
    
    def _add_new_channels_worker(self, image):
        # convert the image to grayscale
        gray = cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
        
        new_channels = []
        if self.add_magnitude_channel:
            new_channels.append(np.sqrt(cv.Sobel(gray,cv.CV_64F,1,0,ksize=7)**2 + cv.Sobel(gray,cv.CV_64F,0,1,ksize=7)**2) )
        
        #if fast_fourier is required, calculate it
        if self.add_fft_channel:
            new_channels.append(20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9))
        
        #if localbinary pattern is required, calculate it
        if self.add_lbp_channel:
            new_channels.append(feature.local_binary_pattern(gray, 3, 6, method='uniform'))

        new_channels = np.stack(new_channels, axis=2) / 255
        return torch.from_numpy(new_channels).to(self.device).float()
        
    def add_new_channels(self, images):
        #copy the input image to avoid modifying the originalu
        images_copied = einops.rearrange(images, "b c h w -> b h w c")
        
        # parallelize over each image in the batch using pool
        new_channels = torch.stack([self._add_new_channels_worker(image) for image in images_copied], dim=0)
        
        # concatenates the new channels to the input image in the channel dimension
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)
        # cast img again to torch tensor and then reshape to (B, C, H, W)
        images_copied = einops.rearrange(images_copied, "b h w c -> b c h w")
        return images_copied
    
    def on_train_start(self):
        return self._on_start()
    
    def on_test_start(self):
        return self._on_start()
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def training_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="test")
    
    def on_train_epoch_end(self):
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
    
    def _step(self, batch, i_batch, phase=None):
        images = batch["image"].to(self.device)
        outs = {
            "phase": phase,
            "labels": batch["is_real"][:, 0].float().to(self.device),
        }
        outs.update(self(images))
        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(input=outs["logits"][:, 0], target=outs["labels"], pos_weight=torch.as_tensor(self.pos_weight, device=self.device))
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        # transfer each tensor to cpu previous to saving them
        for k in outs:
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()
        if phase in {"train", "val"}:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu()), ("learning_rate", self.optimizers().param_groups[0]["lr"])]}, prog_bar=False, logger=True)
        else:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu())]}, prog_bar=False, logger=True)
        # saves the outputs
        self.epoch_outs.append(outs)
        return loss
    
    def _on_start(self):
        with torch.no_grad():
            flops = FlopCountAnalysis(self, torch.randn(1, 3, 224, 224, device=self.device))
            parameters = parameter_count(self)[""]
            self.log_dict({
                "flops": flops.total(),
                "parameters": parameters
                }, prog_bar=True, logger=True)
            
        
    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        self._clear_memory()
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
                self.log_dict({f"{phase}_{k}": v for k, v in metrics.items() if isinstance(v, (torch.Tensor, int, float))}, prog_bar=True, logger=True)
                    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
         
        
if __name__ == "__main__":
    model = BNext4DFR(num_classes=2)
    # runs a dummy forward pass to check if the model is working properly
    model(torch.randn(8, 3, 224, 224))
