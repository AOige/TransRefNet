"""
train.py
--------
Training script for TransRefNet and ConvSegNet (baseline).

Usage — local
-------------
    python train.py --model transrefnet --dataset kvasir --data_root ./data

Usage — Google Colab
--------------------
    !python train.py \
        --model      transrefnet \
        --dataset    clinicdb \
        --data_root  /content/drive/MyDrive/TransRefNet/data \
        --save_dir   /content/drive/MyDrive/TransRefNet/checkpoints \
        --epochs     300 \
        --batch_size 8

Supported models   : transrefnet, convsegnet
Supported datasets : kvasir, clinicdb, isic2018
"""

import os
import time
import datetime
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from model   import ConvSegNet, TransRefNet
from metrics import DiceBCELoss
from dataset import get_loaders, DATASET_NAMES
from utils   import (
    seeding, create_dir, print_and_save,
    epoch_time, calculate_metrics
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TransRefNet or ConvSegNet on a biomedical segmentation dataset."
    )
    p.add_argument("--model",       type=str, default="transrefnet",
                   choices=["transrefnet", "convsegnet"],
                   help="Architecture to train.")
    p.add_argument("--dataset",     type=str, default="kvasir",
                   choices=DATASET_NAMES,
                   help="Dataset identifier.")
    p.add_argument("--data_root",   type=str, required=True,
                   help="Root directory that contains the raw dataset folder.")
    p.add_argument("--save_dir",    type=str, default="./files",
                   help="Directory for checkpoints and logs.")
    p.add_argument("--img_size",    type=int, default=256,
                   help="Square input resolution (default 256).")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--epochs",      type=int, default=300)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int, default=50,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = epoch_jac = epoch_f1 = epoch_rec = epoch_prec = 0.0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss   = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        jac, f1, rec, prec = _batch_metrics(y, y_pred)
        epoch_jac  += jac
        epoch_f1   += f1
        epoch_rec  += rec
        epoch_prec += prec

    n = len(loader)
    return epoch_loss / n, [epoch_jac / n, epoch_f1 / n,
                             epoch_rec / n, epoch_prec / n]


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = epoch_jac = epoch_f1 = epoch_rec = epoch_prec = 0.0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        y_pred    = model(x)
        loss      = loss_fn(y_pred, y)
        epoch_loss += loss.item()

        jac, f1, rec, prec = _batch_metrics(y, y_pred)
        epoch_jac  += jac
        epoch_f1   += f1
        epoch_rec  += rec
        epoch_prec += prec

    n = len(loader)
    return epoch_loss / n, [epoch_jac / n, epoch_f1 / n,
                             epoch_rec / n, epoch_prec / n]


def _batch_metrics(y_true, y_pred):
    """Compute mean metrics over a batch."""
    batch_jac = batch_f1 = batch_rec = batch_prec = []
    for yt, yp in zip(y_true, y_pred):
        scores = calculate_metrics(yt, yp)
        batch_jac.append(scores[0])
        batch_f1.append(scores[1])
        batch_rec.append(scores[2])
        batch_prec.append(scores[3])
    return (np.mean(batch_jac), np.mean(batch_f1),
            np.mean(batch_rec), np.mean(batch_prec))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seeding(args.seed)

    # Paths
    run_name = f"{args.model}_{args.dataset}"
    save_dir = os.path.join(args.save_dir, run_name)
    create_dir(save_dir)

    log_path       = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "best_checkpoint.pth")

    # Log header
    print_and_save(log_path, str(datetime.datetime.now()))
    cfg_str = (
        f"Model:         {args.model}\n"
        f"Dataset:       {args.dataset}\n"
        f"Data root:     {args.data_root}\n"
        f"Image size:    {args.img_size}x{args.img_size}\n"
        f"Batch size:    {args.batch_size}\n"
        f"Epochs:        {args.epochs}\n"
        f"LR:            {args.lr}\n"
        f"Patience:      {args.patience}\n"
        f"Seed:          {args.seed}\n"
    )
    print_and_save(log_path, cfg_str)

    # Data
    img_size = (args.img_size, args.img_size)
    train_loader, val_loader, _ = get_loaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data_str = (
        f"Train batches: {len(train_loader)}\n"
        f"Val batches:   {len(val_loader)}\n"
    )
    print_and_save(log_path, data_str)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_save(log_path, f"Device: {device}\n")

    if args.model == "transrefnet":
        model = TransRefNet()
    else:
        model = ConvSegNet()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )
    loss_fn = DiceBCELoss()

    print_and_save(log_path, "Optimizer: Adam\nLoss: DiceBCE\n")

    # Training loop
    best_f1    = 0.0
    es_count   = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        tr_loss, tr_m   = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_m = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        improved = val_m[1] > best_f1   # F1 is index 1
        if improved:
            best_f1  = val_m[1]
            torch.save(
                {"epoch": epoch + 1, "model_state": model.state_dict(),
                 "val_f1": best_f1, "args": vars(args)},
                checkpoint_path
            )
            es_count = 0
            save_flag = f"  ← best F1 ({best_f1:.4f})  checkpoint saved"
        else:
            es_count += 1
            save_flag = ""

        mins, secs = epoch_time(t0, time.time())
        row = (
            f"Epoch {epoch+1:03d}/{args.epochs}"
            f"  [{mins}m {secs}s]{save_flag}\n"
            f"  Train  loss={tr_loss:.4f}  "
            f"Jac={tr_m[0]:.4f}  F1={tr_m[1]:.4f}  "
            f"Rec={tr_m[2]:.4f}  Prec={tr_m[3]:.4f}\n"
            f"  Val    loss={val_loss:.4f}  "
            f"Jac={val_m[0]:.4f}  F1={val_m[1]:.4f}  "
            f"Rec={val_m[2]:.4f}  Prec={val_m[3]:.4f}\n"
        )
        print_and_save(log_path, row)

        if es_count >= args.patience:
            print_and_save(
                log_path,
                f"Early stopping triggered after {args.patience} epochs "
                f"without improvement.\n"
            )
            break

    print_and_save(log_path, f"\nTraining complete.  Best val F1: {best_f1:.4f}\n")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
