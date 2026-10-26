"""
dataset.py
----------
Unified dataset loader for TransRefNet.

Supported datasets
------------------
  kvasir     : Kvasir-SEG polyp segmentation (1 000 images)
  clinicdb   : CVC-ClinicDB polyp segmentation (612 images)
  isic2018   : ISIC 2018 Task 1 skin-lesion segmentation (2 594 images)

Usage
-----
    from dataset import get_loaders
    train_loader, val_loader, test_loader = get_loaders(
        dataset_name = "kvasir",
        data_root    = "/content/drive/MyDrive/TransRefNet/data",
        batch_size   = 16,
        img_size     = (256, 256),
        num_workers  = 2,
    )
"""

import os
import random
import numpy as np
import cv2
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42


def _seed_worker(worker_id):
    """Ensure reproducible DataLoader workers."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Core dataset class
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Generic paired image–mask segmentation dataset.

    Parameters
    ----------
    images  : list[str]  – absolute paths to RGB images
    masks   : list[str]  – absolute paths to grayscale masks (same order)
    img_size: (W, H)     – target resolution after resize
    augment : bool       – apply training-time augmentations
    """

    _AUGMENT = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(1, 32),
            hole_width_range=(1, 32),
            fill=0, p=0.3
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1, p=0.2
        ),
    ])

    def __init__(self, images, masks, img_size=(256, 256), augment=False):
        assert len(images) == len(masks), "Image / mask count mismatch."
        self.images   = images
        self.masks    = masks
        self.img_size = img_size   # (W, H) for cv2.resize
        self.augment  = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[idx],  cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.images[idx]}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask:  {self.masks[idx]}")

        # Augment before resize so spatial ops are consistent
        if self.augment:
            aug  = self._AUGMENT(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]

        img  = cv2.resize(img,  self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Image: HWC → CHW, normalise to [0, 1]
        img  = np.transpose(img.astype(np.float32), (2, 0, 1)) / 255.0

        # Mask: HW → 1HW, binarise
        mask = np.expand_dims(mask.astype(np.float32), 0) / 255.0
        mask = (mask >= 0.5).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_split(paths, ratios=(0.80, 0.10, 0.10), seed=SEED):
    """
    Randomly split a list into (train, val, test) with fixed seed.
    ratios must sum to 1.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1."
    rng   = random.Random(seed)
    items = list(paths)
    rng.shuffle(items)

    n     = len(items)
    n_tr  = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return items[:n_tr], items[n_tr:n_tr + n_val], items[n_tr + n_val:]


def _paired_split(images, masks, ratios=(0.80, 0.10, 0.10), seed=SEED):
    """Split parallel image and mask lists together."""
    pairs = list(zip(images, masks))
    rng   = random.Random(seed)
    rng.shuffle(pairs)

    n     = len(pairs)
    n_tr  = int(n * ratios[0])
    n_val = int(n * ratios[1])

    tr  = pairs[:n_tr]
    val = pairs[n_tr:n_tr + n_val]
    te  = pairs[n_tr + n_val:]

    def unzip(lst):
        imgs, msks = zip(*lst) if lst else ([], [])
        return list(imgs), list(msks)

    return unzip(tr), unzip(val), unzip(te)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset path discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_dir(root, target_name):
    """
    Walk root looking for a subdirectory named target_name.
    Returns the first match, or None.
    """
    for dirpath, dirnames, _ in os.walk(root):
        if target_name in dirnames:
            return os.path.join(dirpath, target_name)
    return None


def _collect_kvasir(root):
    """
    Expected layout after Kaggle download (debeshjha1/kvasirseg):

        <root>/kvasir-seg/
            images/   *.jpg
            masks/    *.jpg        (same stem as images)

    Searches recursively so double-nested zips
    (kvasir-seg/kvasir-seg/...) are handled automatically.
    """
    img_dir  = _find_dir(root, "images")
    mask_dir = _find_dir(root, "masks")

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not locate 'images' or 'masks' folder under {root}. "
            f"Run download_data.py first."
        )

    images = sorted(glob(os.path.join(img_dir,  "*.jpg")))
    masks  = sorted(glob(os.path.join(mask_dir, "*.jpg")))

    assert len(images) > 0, f"No Kvasir images found under {root}"
    assert len(images) == len(masks), \
        f"Kvasir image/mask count mismatch: {len(images)} vs {len(masks)}"

    return images, masks


def _collect_clinicdb(root):
    """
    Expected layout after Kaggle download (balraj98/cvcclinicdb):

        <root>/CVC-ClinicDB/
            Original/        *.png   (colonoscopy frames)
            Ground Truth/    *.png   (binary masks, same stem)

    Searches recursively to handle any zip nesting depth.
    """
    img_dir  = _find_dir(root, "Original")
    mask_dir = _find_dir(root, "Ground Truth")

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not locate 'Original' or 'Ground Truth' folder under {root}. "
            f"Run download_data.py first."
        )

    images = sorted(glob(os.path.join(img_dir,  "*.png")))
    masks  = sorted(glob(os.path.join(mask_dir, "*.png")))

    assert len(images) > 0, f"No CVC-ClinicDB images found under {root}"
    assert len(images) == len(masks), \
        f"ClinicDB image/mask count mismatch: {len(images)} vs {len(masks)}"

    return images, masks


def _collect_isic2018(root):
    """
    Expected layout after ISIC 2018 download:

        <root>/ISIC2018/
            ISIC2018_Task1-2_Training_Input/
                ISIC_xxxxxxxx.jpg
            ISIC2018_Task1_Training_GroundTruth/
                ISIC_xxxxxxxx_segmentation.png

    We use the training split only (2 594 images) and make our own
    80/10/10 split so all experiments are comparable.
    Searches recursively to handle any zip nesting depth.
    """
    img_dir  = _find_dir(root, "ISIC2018_Task1-2_Training_Input")
    mask_dir = _find_dir(root, "ISIC2018_Task1_Training_GroundTruth")

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not locate ISIC 2018 training folders under {root}. "
            f"Run download_data.py first."
        )

    images_raw = sorted(glob(os.path.join(img_dir, "*.jpg")))

    # Build mask path from image stem
    images, masks = [], []
    for img_path in images_raw:
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, stem + "_segmentation.png")
        if os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)

    assert len(images) > 0, f"No ISIC 2018 images found under {root}"
    return images, masks


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

_COLLECTORS = {
    "kvasir":   _collect_kvasir,
    "clinicdb": _collect_clinicdb,
    "isic2018": _collect_isic2018,
}

DATASET_NAMES = list(_COLLECTORS.keys())


def get_loaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 16,
    img_size: tuple = (256, 256),
    num_workers: int = 2,
    split_ratios: tuple = (0.80, 0.10, 0.10),
    seed: int = SEED,
):
    """
    Build DataLoaders for the requested dataset.

    Parameters
    ----------
    dataset_name : str   – one of DATASET_NAMES
    data_root    : str   – root directory containing the raw dataset folder
    batch_size   : int
    img_size     : (W, H)
    num_workers  : int
    split_ratios : (train, val, test)  – must sum to 1
    seed         : int   – random seed for reproducibility

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _COLLECTORS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {DATASET_NAMES}"
        )

    images, masks = _COLLECTORS[dataset_name](data_root)
    (tr_imgs, tr_masks), (val_imgs, val_masks), (te_imgs, te_masks) = \
        _paired_split(images, masks, ratios=split_ratios, seed=seed)

    print(
        f"\n[Dataset] {dataset_name.upper()}"
        f"  |  total={len(images)}"
        f"  |  train={len(tr_imgs)}"
        f"  |  val={len(val_imgs)}"
        f"  |  test={len(te_imgs)}\n"
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        SegmentationDataset(tr_imgs,  tr_masks,  img_size, augment=True),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=g,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_imgs, val_masks, img_size, augment=False),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        SegmentationDataset(te_imgs,  te_masks,  img_size, augment=False),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_dataset_stats(dataset_name: str, data_root: str):
    """Return (n_train, n_val, n_test) for a dataset without building loaders."""
    images, masks = _COLLECTORS[dataset_name.lower()](data_root)
    (a, _), (b, _), (c, _) = _paired_split(images, masks)
    return len(a), len(b), len(c)
