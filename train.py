#!/usr/bin/env python3
from pprint import pprint
import argparse
from collections import OrderedDict
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchmetrics.functional.classification import accuracy
import lightning as L
from lightning.pytorch.loggers import WandbLogger
    
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset

import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate
import random
import numpy as np

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/bin_caddm_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default=None)
    parser.add_argument('--seed', type=str, help='The seed to use.', default=5)
    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    opt_state_dict = {}
    opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    opt_state_dict['state'] = OrderedDict()
    for k, v in opt.state_dict()['state'].items():
        opt_state_dict['state'][k] = {}
        opt_state_dict['state'][k]['step'] = v['step']
        if 'exp_avg' in v:
            opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
        if 'exp_avg_sq' in v:
            opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}.pkl')


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module."+k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1
    return net, opt, base_epoch
            
def train():
    # def training_epoch(net, dataloader, device, accumulation_batches, metrics=None, epoch=None):
    #     net.train()
    #     if metrics is None:
    #         metrics = pd.DataFrame()
    #     progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    #     for i_batch, batch in progress_bar:
    #         # lr = update_learning_rate(epoch)
    #         # for param_group in optimizer.param_groups:
    #         #     param_group['lr'] = lr
    #         images = batch["image"].to(device)
    #         labels = batch["is_real"][:, 0].float().to(device)

    #         with torch.cuda.amp.autocast(enabled=True):
    #             labels_pred = net(images)
    #             # TODO handle class imbalance
    #             loss = F.binary_cross_entropy_with_logits(input=labels_pred, target=labels)
    #         # scaler.scale(loss).backward()
    #         loss.backward()
            
    #         # gradient accumulation
    #         if ((i_batch + 1) % accumulation_batches) == 0:
    #             # scaler.unscale_(optimizer)
    #             # nn.utils.clip_grad_value_(net.parameters(), 5)
    #             # scaler.step(optimizer)
    #             optimizer.step()
    #             # scaler.update()
    #             optimizer.zero_grad()

    #         # metrics update
    #         with torch.no_grad():
    #             metrics = pd.concat([
    #                 metrics,
    #                 pd.DataFrame({
    #                 "epoch": [epoch],
    #                 "phase": ["train"],
    #                 "batch": [i_batch],
    #                 "loss": [loss.cpu().item()],
    #                 "accuracy": [accuracy(preds=labels_pred.cpu(), target=labels.cpu(), task="binary").item()],
    #             })]
    #             ) 
    #         metrics_per_epoch = metrics[(metrics['epoch'] == epoch) & ((metrics['phase'] == 'train'))].drop("phase", axis="columns").mean()
    #         progress_bar.set_description(f"train epoch {epoch}: acc={metrics_per_epoch['accuracy'] * 100:.1f}%, loss={metrics_per_epoch['loss']:.3f}")
            
    # def val_epoch(net, dataloader, device, metrics=None, epoch=None):
    #     net.eval()
    #     if metrics is None:
    #         metrics = pd.DataFrame()
    #     progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    #     for i_batch, batch in progress_bar:
    #         # lr = update_learning_rate(epoch)
    #         # for param_group in optimizer.param_groups:
    #         #     param_group['lr'] = lr
    #         images = batch["image"].to(device)
    #         labels = batch["is_real"][:, 0].float().to(device)

    #         with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
    #             labels_pred = net(images)
    #             loss = F.binary_cross_entropy_with_logits(input=labels_pred, target=labels)

    #         # metrics update
    #         with torch.no_grad():
    #             metrics = pd.concat([
    #                 metrics,
    #                 pd.DataFrame({
    #                 "epoch": [epoch],
    #                 "phase": ["val"],
    #                 "batch": [i_batch],
    #                 "loss": [loss.cpu().item()],
    #                 "accuracy": [accuracy(preds=labels_pred.cpu(), target=labels.cpu(), task="binary").item()],
    #             })]
    #             ) 
    #         metrics_per_epoch = metrics[(metrics['epoch'] == epoch) & ((metrics['phase'] == 'val'))].drop("phase", axis="columns").mean()
    #         progress_bar.set_description(f"val epoch {epoch}: acc={metrics_per_epoch['accuracy'] * 100:.1f}%, loss={metrics_per_epoch['loss']:.3f}")
        
    args = args_func()
    
    # preliminary setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # get data
    if cfg["dataset"]["name"] == "coco_fake":
        print(f"Load COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}")
        # train_dataset = DeepfakeDataset('train', cfg)
        train_dataset = COCOFakeDataset(coco2014_path=cfg["dataset"]["coco2014_path"], coco_fake_path=cfg["dataset"]["coco_fake_path"], split="train", mode="single", resolution=cfg["train"]["resolution"])
        val_dataset = COCOFakeDataset(coco2014_path=cfg["dataset"]["coco2014_path"], coco_fake_path=cfg["dataset"]["coco_fake_path"], split="val", mode="single", resolution=cfg["train"]["resolution"])
        test_dataset = val_dataset
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Load DFFD dataset from {cfg['dataset']['dffd_path']}")
        train_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="train", resolution=cfg["train"]["resolution"])
        val_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="val", resolution=cfg["train"]["resolution"])
        test_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="test", resolution=cfg["train"]["resolution"])
    
    # loads the dataloaders
    num_workers = os.cpu_count() // 2
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=num_workers,
                              )
    val_loader = DataLoader(val_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=False, num_workers=num_workers,
                              )
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=False, num_workers=num_workers,
                              )

    # init model
    positive_samples = sum([item["is_real"] for item in train_dataset.items])
    negative_samples = len(train_dataset) - positive_samples
    net = model.get(labels=cfg["dataset"]["labels"], backbone=cfg['model']['backbone'], add_magnitude_channel=cfg['model']['add_magnitude_channel'], add_fft_channel=cfg['model']['add_fft_channel'], add_lbp_channel=cfg['model']['add_lbp_channel'], pos_weight=negative_samples / positive_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # start training
    date = datetime.now().strftime('%Y%m%d_%H%M')
    logger = WandbLogger(project="DFAD_CVPRW24", name=cfg["dataset"]["name"] + f"_{date}", log_model=False)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in str(device) else "cpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.,
        accumulate_grad_batches=cfg["train"]["accumulation_batches"],
        limit_train_batches=cfg["train"]["limit_train_batches"], 
        limit_val_batches=cfg["train"]["limit_val_batches"],
        max_epochs=cfg['train']["epoch_num"],
        logger=logger)
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=net, dataloaders=test_loader)


if __name__ == "__main__":
    train()
