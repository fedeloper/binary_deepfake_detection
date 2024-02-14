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
            
if __name__ == "__main__":
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)
    
    # preliminary setup
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # get data
    if cfg["dataset"]["name"] == "coco_fake":
        print(f"Load COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}")
        test_dataset = COCOFakeDataset(coco2014_path=cfg["dataset"]["coco2014_path"], coco_fake_path=cfg["dataset"]["coco_fake_path"], split="val", mode="single", resolution=cfg["test"]["resolution"])
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Load DFFD dataset from {cfg['dataset']['dffd_path']}")
        test_dataset = DFFDDataset(dataset_path=cfg["dataset"]["dffd_path"], split="test", resolution=cfg["test"]["resolution"])
    
    # loads the dataloaders
    num_workers = os.cpu_count() // 2
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['test']['batch_size'],
                              shuffle=False, num_workers=num_workers,
                              )

    # init model
    net = model.CADDM.load_from_checkpoint(cfg["test"]["checkpoint_path"])
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
        precision="16-mixed" if cfg["train"]["mixed_precision"] else 32,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.,
        accumulate_grad_batches=cfg["test"]["accumulation_batches"],
        limit_test_batches=cfg["test"]["limit_test_batches"],
        max_epochs=cfg['test']["epoch_num"],
        logger=logger)
    trainer.test(model=net, dataloaders=test_loader)
