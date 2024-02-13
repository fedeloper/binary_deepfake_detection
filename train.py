#!/usr/bin/env python3
from pprint import pprint
import argparse
from collections import OrderedDict
import os
from os.path import join
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
    args = parser.parse_args()
    return args
            
def train():        
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)
    
    # preliminary setup
    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    torch.set_float32_matmul_precision("medium")

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
    project = "DFAD_CVPRW24"
    run = cfg["dataset"]["name"] + f"_{date}"
    logger = WandbLogger(project=project, name=run, id=run, log_model=False)
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
        callbacks=[L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_acc", save_top_k=1, mode="max",
            filename='{epoch}-{train_acc:.2f}-{val_acc:.2f}',
            )],
        logger=logger)
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model=net, dataloaders=test_loader)


if __name__ == "__main__":
    train()
