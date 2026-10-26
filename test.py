"""
test.py
-------
Evaluate a trained TransRefNet (or ConvSegNet) checkpoint on the held-out
test split of a dataset.

Usage — local
-------------
    python test.py \
        --checkpoint ./files/transrefnet_kvasir/best_checkpoint.pth \
        --dataset    kvasir \
        --data_root  ./data

Usage — Google Colab
--------------------
    !python test.py \
        --checkpoint /content/drive/MyDrive/TransRefNet/checkpoints/transrefnet_kvasir/best_checkpoint.pth \
        --dataset    kvasir \
        --data_root  /content/drive/MyDrive/TransRefNet/data \
        --save_preds /content/drive/MyDrive/TransRefNet/predictions/kvasir

Outputs
-------
  * Per-sample CSV  : <results_dir>/test_results.csv
  * Summary metrics : printed to stdout and saved to <results_dir>/test_summary.txt
  * Optional predicted masks (--save_preds DIR)
"""

import os
import csv
import argparse

import numpy as np
import cv2
import torch

from model   import ConvSegNet, TransRefNet
from dataset import get_loaders, DATASET_NAMES
from utils   import seeding, create_dir, print_and_save, calculate_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a TransRefNet or ConvSegNet checkpoint."
    )
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to the .pth checkpoint saved by train.py.")
    p.add_argument("--dataset",     type=str, required=True,
                   choices=DATASET_NAMES,
                   help="Dataset to evaluate on.")
    p.add_argument("--data_root",   type=str, required=True,
                   help="Root directory containing the raw dataset folder.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Where to save CSV and summary (default: same dir as checkpoint).")
    p.add_argument("--save_preds",  type=str, default=None,
                   help="If set, save predicted binary masks to this directory.")
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, save_dir=None):
    """
    Run inference on the loader and collect per-sample metrics.

    Returns
    -------
    results : list[dict]  –  one dict per sample with keys
                             jac, f1, recall, precision
    """
    model.eval()
    results = []
    global_idx = 0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        y_pred = model(x)

        for i in range(x.shape[0]):
            jac, f1, rec, prec = calculate_metrics(y[i], y_pred[i])
            results.append({
                "idx":       global_idx,
                "jaccard":   jac,
                "f1":        f1,
                "recall":    rec,
                "precision": prec,
            })

            # Optionally save prediction mask
            if save_dir is not None:
                pred_bin = (torch.sigmoid(y_pred[i]) > 0.5).cpu().numpy()
                pred_bin = (pred_bin.squeeze() * 255).astype(np.uint8)
                out_path = os.path.join(save_dir, f"pred_{global_idx:04d}.png")
                cv2.imwrite(out_path, pred_bin)

            global_idx += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seeding(args.seed)

    # Determine results directory
    if args.results_dir is None:
        args.results_dir = os.path.dirname(args.checkpoint)
    create_dir(args.results_dir)

    if args.save_preds:
        create_dir(args.save_preds)

    summary_path = os.path.join(args.results_dir, "test_summary.txt")
    csv_path     = os.path.join(args.results_dir, "test_results.csv")

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    # Determine architecture from checkpoint metadata; fall back to CLI arg
    model_name = saved_args.get("model", "transrefnet")
    if model_name == "convsegnet":
        model = ConvSegNet()
    else:
        model = TransRefNet()

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print_and_save(summary_path,
                   f"Checkpoint : {args.checkpoint}")
    print_and_save(summary_path,
                   f"Architecture: {model_name}")
    print_and_save(summary_path,
                   f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print_and_save(summary_path,
                   f"Best val F1 (train): {ckpt.get('val_f1', 'N/A'):.4f}\n")

    # Data loaders — only test split is used
    img_size = (args.img_size, args.img_size)
    _, _, test_loader = get_loaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Run evaluation
    print_and_save(summary_path, f"Evaluating on {args.dataset.upper()} test set...")
    results = evaluate(model, test_loader, device, save_dir=args.save_preds)

    # Write per-sample CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "jaccard", "f1",
                                                "recall", "precision"])
        writer.writeheader()
        writer.writerows(results)

    # Compute aggregate statistics
    jacs   = [r["jaccard"]   for r in results]
    f1s    = [r["f1"]        for r in results]
    recs   = [r["recall"]    for r in results]
    precs  = [r["precision"] for r in results]

    summary = (
        f"\n{'─'*50}\n"
        f"  TEST RESULTS  —  {args.dataset.upper()}\n"
        f"{'─'*50}\n"
        f"  Samples   : {len(results)}\n"
        f"  Jaccard   : {np.mean(jacs):.4f}  ±  {np.std(jacs):.4f}\n"
        f"  F1 (Dice) : {np.mean(f1s):.4f}  ±  {np.std(f1s):.4f}\n"
        f"  Recall    : {np.mean(recs):.4f}  ±  {np.std(recs):.4f}\n"
        f"  Precision : {np.mean(precs):.4f}  ±  {np.std(precs):.4f}\n"
        f"{'─'*50}\n"
    )
    print_and_save(summary_path, summary)

    if args.save_preds:
        print_and_save(summary_path,
                       f"Predicted masks saved to: {args.save_preds}")

    print(f"\nPer-sample CSV : {csv_path}")
    print(f"Summary        : {summary_path}")


if __name__ == "__main__":
    main()
