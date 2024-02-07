#!/usr/bin/env python3
from pprint import pprint
import argparse
from collections import OrderedDict
import os
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchmetrics.functional.classification import accuracy
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
    def training_epoch(net, dataloader, device, accumulation_batches, metrics=None, epoch=None):
        net.train()
        if metrics is None:
            metrics = pd.DataFrame()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i_batch, batch in progress_bar:
            # lr = update_learning_rate(epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            images = batch["image"].to(device)
            labels = batch["is_real"][:, 0].float().to(device)

            with torch.cuda.amp.autocast(enabled=True):
                labels_pred = net(images)
                # TODO handle class imbalance
                loss = F.binary_cross_entropy_with_logits(input=labels_pred, target=labels)
            # scaler.scale(loss).backward()
            loss.backward()
            
            # gradient accumulation
            if ((i_batch + 1) % accumulation_batches) == 0:
                # scaler.unscale_(optimizer)
                # nn.utils.clip_grad_value_(net.parameters(), 5)
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                optimizer.zero_grad()

            # metrics update
            with torch.no_grad():
                metrics = pd.concat([
                    metrics,
                    pd.DataFrame({
                    "epoch": [epoch],
                    "phase": ["train"],
                    "batch": [i_batch],
                    "loss": [loss.cpu().item()],
                    "accuracy": [accuracy(preds=labels_pred.cpu(), target=labels.cpu(), task="binary").item()],
                })]
                ) 
            metrics_per_epoch = metrics[(metrics['epoch'] == epoch) & ((metrics['phase'] == 'train'))].drop("phase", axis="columns").mean()
            progress_bar.set_description(f"train epoch {epoch}: acc={metrics_per_epoch['accuracy'] * 100:.1f}%, loss={metrics_per_epoch['loss']:.3f}")
            
    def val_epoch(net, dataloader, device, metrics=None, epoch=None):
        net.eval()
        if metrics is None:
            metrics = pd.DataFrame()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i_batch, batch in progress_bar:
            # lr = update_learning_rate(epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            images = batch["image"].to(device)
            labels = batch["is_real"][:, 0].float().to(device)

            with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
                labels_pred = net(images)
                loss = F.binary_cross_entropy_with_logits(input=labels_pred, target=labels)

            # metrics update
            with torch.no_grad():
                metrics = pd.concat([
                    metrics,
                    pd.DataFrame({
                    "epoch": [epoch],
                    "phase": ["val"],
                    "batch": [i_batch],
                    "loss": [loss.cpu().item()],
                    "accuracy": [accuracy(preds=labels_pred.cpu(), target=labels.cpu(), task="binary").item()],
                })]
                ) 
            metrics_per_epoch = metrics[(metrics['epoch'] == epoch) & ((metrics['phase'] == 'val'))].drop("phase", axis="columns").mean()
            progress_bar.set_description(f"val epoch {epoch}: acc={metrics_per_epoch['accuracy'] * 100:.1f}%, loss={metrics_per_epoch['loss']:.3f}")
        
    args = args_func()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # init model.
    net = model.get(labels=cfg["dataset"]["labels"], backbone=cfg['model']['backbone'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # net = nn.DataParallel(net)

    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    # if args.ckpt:
    #     net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)

    # get training data
    if cfg["dataset"]["name"] == "coco_fake":
        print(f"Load COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}")
        # train_dataset = DeepfakeDataset('train', cfg)
        train_dataset = COCOFakeDataset(coco2014_path=cfg["dataset"]["coco2014_path"], coco_fake_path=cfg["dataset"]["coco_fake_path"], split="train", mode="single", resolution=cfg["train"]["resolution"])
        # splits the training set into training and validation
        # TODO discuss about train/val split
        real_indices = [i for i in range(len(train_dataset)) if train_dataset.items[i]["is_real"]]
        fake_indices = [i for i in range(len(train_dataset)) if not train_dataset.items[i]["is_real"]]
        assert sorted(real_indices + fake_indices) == list(range(len(train_dataset))), "indices mismatch"
        random.shuffle(real_indices)
        random.shuffle(fake_indices)
        train_dataset = Subset(train_dataset, real_indices[:int(len(real_indices) * 0.8)] + fake_indices[:int(len(fake_indices) * 0.8)])
        val_dataset = Subset(train_dataset, real_indices[int(len(real_indices) * 0.8):] + fake_indices[int(len(fake_indices) * 0.8):])
        # use the val set as test
        test_dataset = COCOFakeDataset(coco2014_path=cfg["dataset"]["coco2014_path"], coco_fake_path=cfg["dataset"]["coco_fake_path"], split="val", mode="single", resolution=cfg["train"]["resolution"])
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Load DFFD dataset from {cfg['dataset']['dffd_path']}")
        train_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="train", resolution=cfg["train"]["resolution"])
        val_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="val", resolution=cfg["train"]["resolution"])
        test_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="test", resolution=cfg["train"]["resolution"])
    # eventually reduce the size of the training set
    assert 0 < cfg["train"]["limit_train_batches"] <= 1.0, f"got {cfg['train']['limit_train_batches']}"
    train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(len(train_dataset) * cfg["train"]["limit_train_batches"])])
    
    # loads the dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=2,
                              )
    val_loader = DataLoader(val_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=False, num_workers=2,
                              )
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=False, num_workers=2,
                              )

    # start trining.
    metrics = pd.DataFrame()
    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        training_epoch(net=net, dataloader=train_loader, device=device, accumulation_batches=cfg["train"]["accumulation_batches"], epoch=epoch, metrics=metrics)
        val_epoch(net=net, dataloader=val_loader, device=device, epoch=epoch, metrics=metrics)

        # save_checkpoint(net, optimizer,
        #                 cfg['model']['save_path'],
        #                 epoch)


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
