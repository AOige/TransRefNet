# TransRefNet: Transformer-Augmented Refinement Network for Biomedical Image Segmentation

TransRefNet extends the ConvSegNet architecture by introducing a lightweight **Transformer Bridge** at the encoder bottleneck.  It captures long-range spatial dependencies that purely convolutional models cannot model, while keeping the proven Context Feature Refinement (CFR) decoder unchanged — enabling a clean ablation that isolates the contribution of global self-attention.

---

## Architecture Overview

```
Input (256×256)
    │
    ▼
ResNet50 Encoder
    │
    ├─ s1  [B, 64,  128, 128]  ──────────────────────────────┐
    ├─ s2  [B, 64,   64,  64]  ──────────────────────────┐   │
    ├─ s3  [B, 64,   32,  32]  ──────────────────────┐   │   │
    └─ s4  [B, 64,   16,  16]                        │   │   │
              │                                      │   │   │
              ▼                                      │   │   │
    TransformerBridge                                │   │   │
      2-layer Pre-LN Transformer                     │   │   │
      8-head MHSA over 256 tokens                    │   │   │
      Factorised 2-D learned positional embs         │   │   │
      Residual connection                            │   │   │
              │                                      │   │   │
              ▼                                      │   │   │
    CFR Decoder (U-Net style skip connections) ◄─────┘   │   │
      CFR-1: cat(up(s4), s3)  →  [B,64,32,32]            │   │
      CFR-2: cat(up(x),  s2)  →  [B,64,64,64]   ◄────────┘   │
      CFR-3: cat(up(x),  s1)  →  [B,64,128,128] ◄────────────┘
      CFR-4: cat(up(x),  s0)  →  [B,64,256,256]
              │
              ▼
    1×1 Conv → Output mask (256×256)
```

Each **CFR block** applies four parallel convolutions (1×1, 3×3, 7×7, 11×11) and concatenates them, providing multi-scale local context at every decoder stage.

The **only architectural change** versus ConvSegNet is the TransformerBridge inserted between the encoder and the decoder.  All other components are identical, making comparisons between the two models a direct measure of global self-attention's impact.

---

## Repository Structure

```
TransRefNet/
├── model.py            # ConvSegNet + TransformerBridge + TransRefNet
├── dataset.py          # Unified loader for all four datasets
├── train.py            # Training script (argparse, early stopping, LR schedule)
├── test.py             # Evaluation script — per-sample CSV + aggregate metrics
├── metrics.py          # DiceLoss and DiceBCELoss
├── utils.py            # Seeding, logging, metric calculation
├── download_data.py    # Kaggle-based dataset download helper
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Datasets

| Dataset | Task | Images | Split (80/10/10) |
|---------|------|--------|-----------------|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | Polyp segmentation | 1 000 | 800 / 100 / 100 |
| [CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) | Polyp segmentation | 612 | 489 / 61 / 62 |
| [ISIC 2018 Task 1](https://challenge.isic-archive.com/landing/2018/) | Skin lesion segmentation | 2 594 | 2 075 / 259 / 260 |

All splits are generated with `seed=42` and are consistent across ConvSegNet and TransRefNet runs.

---

## Setup

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/TransRefNet.git
cd TransRefNet
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure Kaggle API

Download your `kaggle.json` from **kaggle.com → Account → Create New API Token** and place it at `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 4 — Download datasets

```bash
# Download all three datasets
python download_data.py --data_root ./data --dataset all

# Or one at a time
python download_data.py --data_root ./data --dataset kvasir
python download_data.py --data_root ./data --dataset clinicdb
python download_data.py --data_root ./data --dataset isic2018
```

---

## Google Colab Quick-Start

Open a new Colab notebook, mount your Drive, then run the following cells:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/TransRefNet"
```

```python
# Cell 2 — Clone repo and install dependencies
import os
os.chdir("/content")
!git clone https://github.com/<your-username>/TransRefNet.git
os.chdir("/content/TransRefNet")
!pip install -q albumentations ptflops kaggle
```

```python
# Cell 3 — Set up Kaggle credentials
# Upload kaggle.json via the Files panel (left sidebar) first
from download_data import setup_kaggle, download_all
setup_kaggle("/content/kaggle.json")
download_all(f"{BASE}/data")
```

```python
# Cell 4 — Train TransRefNet on Kvasir-SEG
!python train.py \
    --model      transrefnet \
    --dataset    kvasir \
    --data_root  {BASE}/data \
    --save_dir   {BASE}/checkpoints \
    --epochs     300 \
    --batch_size 8
```

```python
# Cell 5 — Train the ConvSegNet baseline for comparison
!python train.py \
    --model      convsegnet \
    --dataset    kvasir \
    --data_root  {BASE}/data \
    --save_dir   {BASE}/checkpoints \
    --epochs     300 \
    --batch_size 8
```

```python
# Cell 6 — Evaluate on the test split
!python test.py \
    --checkpoint {BASE}/checkpoints/transrefnet_kvasir/best_checkpoint.pth \
    --dataset    kvasir \
    --data_root  {BASE}/data \
    --save_preds {BASE}/predictions/kvasir
```

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | Dice + BCE (equal weight) |
| Input resolution | 256 × 256 |
| Batch size | 16 (local) / 8 (Colab free tier) |
| Max epochs | 300 |
| Early stopping | 50 epochs without val-F1 improvement |
| Seed | 42 |

### Data Augmentation (training only)

- Random rotation ±35°
- Horizontal flip
- Vertical flip
- Coarse dropout (10 patches, 32×32)
- Colour jitter (brightness, contrast, saturation, hue)

---

## Evaluation Metrics

Four standard metrics are reported for every dataset:

| Metric | Formula |
|--------|---------|
| **Jaccard (IoU)** | TP / (TP + FP + FN) |
| **Dice (F1)** | 2·TP / (2·TP + FP + FN) |
| **Recall** | TP / (TP + FN) |
| **Precision** | TP / (TP + FP) |

---

## Running All Experiments

```bash
# Run all six experiments (3 datasets × 2 models)
for MODEL in transrefnet convsegnet; do
  for DATASET in kvasir clinicdb isic2018; do
    python train.py \
      --model      $MODEL \
      --dataset    $DATASET \
      --data_root  ./data \
      --save_dir   ./checkpoints \
      --epochs     300
  done
done
```

Evaluate all:

```bash
for MODEL in transrefnet convsegnet; do
  for DATASET in kvasir clinicdb isic2018; do
    python test.py \
      --checkpoint ./checkpoints/${MODEL}_${DATASET}/best_checkpoint.pth \
      --dataset    $DATASET \
      --data_root  ./data \
      --results_dir ./results/${MODEL}_${DATASET}
  done
done
```

---

## Model Complexity

Run the built-in complexity check (requires `ptflops`):

```bash
python model.py
```

Sample output:

```
ConvSegNet (baseline)
  FLOPs  : ~X GMac
  Params : ~Y M

TransRefNet (proposed)
  FLOPs  : ~X' GMac
  Params : ~Y' M

ConvSegNet output:  (2, 1, 256, 256)
TransRefNet output: (2, 1, 256, 256)
Shape check: PASSED
```

---

## Expected Data Directory Layout

After downloading and preprocessing, `./data` should look like:

```
data/
├── kvasir-seg/
│   ├── images/           *.jpg
│   └── masks/            *.jpg
│
├── CVC-ClinicDB/
│   ├── Original/         *.png
│   └── Ground Truth/     *.png
│
└── ISIC2018/
    ├── ISIC2018_Task1-2_Training_Input/        *.jpg
    └── ISIC2018_Task1_Training_GroundTruth/    *_segmentation.png
```

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{ige2027transrefnet,
  title     = {TransRefNet: Transformer-Augmented Refinement Network for Biomedical Image Segmentation},
  author    = {Ige, Ayokunle},
  booktitle = {Proceedings of the Machine Vision Applications (MVA)},
  year      = {2027},
  address   = {Fukuoka, Japan}
}
```
