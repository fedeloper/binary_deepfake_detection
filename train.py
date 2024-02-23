from pprint import pprint
import argparse
import os
from datetime import datetime
import random
import numpy as np
import gc

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from cifake_dataset import CIFAKEDataset

from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset

import model
from lib.util import load_config



def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/ablation_baseline.cfg",
    )
    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
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
        print(
            f"Loading COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
        )
        # train_dataset = DeepfakeDataset('train', cfg)
        train_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="train",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Loading DFFD dataset from {cfg['dataset']['dffd_path']}")
        train_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        train_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["train"]["resolution"],
        )

    # loads the dataloaders
    num_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # init model
    positive_samples = sum([item["is_real"] for item in train_dataset.items])
    negative_samples = len(train_dataset) - positive_samples
    net = model.BNext4DFR(
        num_classes=cfg["dataset"]["labels"],
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        add_magnitude_channel=cfg["model"]["add_magnitude_channel"],
        add_fft_channel=cfg["model"]["add_fft_channel"],
        add_lbp_channel=cfg["model"]["add_lbp_channel"],
        pos_weight=negative_samples / positive_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # start training
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "DFAD_CVPRW24"
    run_label = args.cfg.split("/")[-1].split(".")[0]
    run = cfg["dataset"]["name"] + f"_{date}_{run_label}"
    logger = WandbLogger(project=project, name=run, id=run, log_model=False)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in str(device) else "cpu",
        devices=1,
        precision="16-mixed" if cfg["train"]["mixed_precision"] else 32,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg["train"]["accumulation_batches"],
        limit_train_batches=cfg["train"]["limit_train_batches"],
        limit_val_batches=cfg["train"]["limit_val_batches"],
        max_epochs=cfg["train"]["epoch_num"],
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                filename= cfg["dataset"]["name"] + "_" + cfg["model"]["backbone"] + "_{epoch}-{train_acc:.2f}-{val_acc:.2f}",
            )
        ],
        logger=logger,
    )
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
