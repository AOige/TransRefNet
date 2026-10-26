"""
download_data.py
----------------
Downloads all four TransRefNet datasets to a common data root directory.

Requirements
------------
  pip install kaggle

  Place your Kaggle API credentials at  ~/.kaggle/kaggle.json
  (or upload kaggle.json in Colab and run setup_kaggle() below).

Usage — local
-------------
    python download_data.py --data_root ./data --dataset all

Usage — Google Colab (paste into a cell)
-----------------------------------------
    from download_data import setup_kaggle, download_all
    setup_kaggle("/content/kaggle.json")          # path where you uploaded it
    download_all("/content/drive/MyDrive/TransRefNet/data")

Individual downloads
--------------------
    python download_data.py --data_root ./data --dataset kvasir
    python download_data.py --data_root ./data --dataset clinicdb
    python download_data.py --data_root ./data --dataset isic2018
    python download_data.py --data_root ./data --dataset dsb2018
"""

import os
import shutil
import zipfile
import argparse
import subprocess


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_kaggle(json_path: str):
    """
    Copy kaggle.json to ~/.kaggle/ and set correct permissions.
    Needed in Colab where the home directory is ephemeral.

    Parameters
    ----------
    json_path : str  Path to the uploaded kaggle.json file.
                     In Colab: upload via the Files panel, then pass the path.
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest = os.path.join(kaggle_dir, "kaggle.json")
    shutil.copy(json_path, dest)
    os.chmod(dest, 0o600)
    print(f"kaggle.json installed at {dest}")


def _run(cmd: str, cwd: str = None):
    """Run a shell command and stream output."""
    print(f"\n$ {cmd}\n")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{cmd}"
        )


def _unzip(zip_path: str, dest_dir: str):
    """Unzip and remove the archive."""
    print(f"Extracting {os.path.basename(zip_path)} → {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)
    print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset downloaders
# ─────────────────────────────────────────────────────────────────────────────

def download_kvasir(data_root: str):
    """
    Kvasir-SEG — 1 000 polyp images + masks
    Kaggle dataset: debeshjha1/kvasirseg
    Final layout: <data_root>/kvasir-seg/{images,masks}/
    """
    out_dir = os.path.join(data_root, "kvasir-seg")
    if _already_downloaded(out_dir, min_files=2000):
        print("[kvasir] Already downloaded, skipping.")
        return

    os.makedirs(data_root, exist_ok=True)
    _run(
        f"kaggle datasets download -d debeshjha1/kvasirseg "
        f"--path {data_root} --unzip",
        cwd=data_root
    )
    # Kaggle unzips into kvasir-seg/ directly
    print(f"[kvasir] Saved to: {out_dir}")


def download_clinicdb(data_root: str):
    """
    CVC-ClinicDB — 612 colonoscopy frames + masks
    Kaggle dataset: balraj98/cvcclinicdb
    Final layout: <data_root>/CVC-ClinicDB/{Original,Ground Truth}/
    """
    out_dir = os.path.join(data_root, "CVC-ClinicDB")
    if _already_downloaded(out_dir, min_files=1224):
        print("[clinicdb] Already downloaded, skipping.")
        return

    os.makedirs(data_root, exist_ok=True)
    _run(
        f"kaggle datasets download -d balraj98/cvcclinicdb "
        f"--path {data_root} --unzip",
        cwd=data_root
    )
    print(f"[clinicdb] Saved to: {out_dir}")


def download_isic2018(data_root: str):
    """
    ISIC 2018 Task 1 — 2 594 skin-lesion images + masks
    Kaggle dataset: shonenkov/isic2018

    Note: If this Kaggle slug changes, check https://www.kaggle.com/datasets
    and search for 'ISIC 2018 Task 1'.

    Final layout:
        <data_root>/ISIC2018/
            ISIC2018_Task1-2_Training_Input/       *.jpg
            ISIC2018_Task1_Training_GroundTruth/   *_segmentation.png
    """
    out_dir = os.path.join(data_root, "ISIC2018")
    if _already_downloaded(out_dir, min_files=5000):
        print("[isic2018] Already downloaded, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    # Primary source: official ISIC 2018 Kaggle mirror
    _run(
        f"kaggle datasets download -d shonenkov/isic2018 "
        f"--path {out_dir} --unzip",
        cwd=out_dir
    )
    print(f"[isic2018] Saved to: {out_dir}")


def download_dsb2018(data_root: str):
    """
    2018 Data Science Bowl — ~670 microscopy nuclei samples
    Kaggle competition: data-science-bowl-2018

    Note: Requires accepting the competition rules on Kaggle first.
    Visit https://www.kaggle.com/c/data-science-bowl-2018 and click
    'Late Submission' to accept terms, then run this download.

    Final layout:
        <data_root>/stage1_train/{sample_id}/{images,masks}/

    After downloading, run:
        from dataset import prepare_dsb2018
        prepare_dsb2018("<data_root>")
    to merge per-nucleus masks into single binary masks.
    """
    out_dir = os.path.join(data_root, "stage1_train")
    if _already_downloaded(out_dir, min_files=500):
        print("[dsb2018] Already downloaded, skipping.")
        return

    os.makedirs(data_root, exist_ok=True)
    _run(
        f"kaggle competitions download -c data-science-bowl-2018 "
        f"--path {data_root}",
        cwd=data_root
    )

    # The competition download is a zip of zips
    zip_path = os.path.join(data_root, "data-science-bowl-2018.zip")
    if os.path.exists(zip_path):
        _unzip(zip_path, data_root)

    # Unzip stage1_train if it's still zipped
    stage1_zip = os.path.join(data_root, "stage1_train.zip")
    if os.path.exists(stage1_zip):
        _unzip(stage1_zip, data_root)

    print(f"[dsb2018] Saved to: {out_dir}")
    print(
        "\n[dsb2018] Next step: run prepare_dsb2018() to merge nucleus masks.\n"
        "  from dataset import prepare_dsb2018\n"
        f"  prepare_dsb2018('{data_root}')\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _already_downloaded(directory: str, min_files: int = 1) -> bool:
    """Return True if directory exists and contains at least min_files files."""
    if not os.path.isdir(directory):
        return False
    count = sum(len(files) for _, _, files in os.walk(directory))
    return count >= min_files


_DOWNLOADERS = {
    "kvasir":   download_kvasir,
    "clinicdb": download_clinicdb,
    "isic2018": download_isic2018,
    "dsb2018":  download_dsb2018,
}


def download_all(data_root: str):
    """Download all four datasets to data_root."""
    for name, fn in _DOWNLOADERS.items():
        print(f"\n{'='*55}")
        print(f"  Downloading: {name.upper()}")
        print(f"{'='*55}")
        try:
            fn(data_root)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download TransRefNet datasets via the Kaggle API."
    )
    p.add_argument(
        "--data_root", type=str, required=True,
        help="Directory to save datasets."
    )
    p.add_argument(
        "--dataset", type=str, default="all",
        choices=list(_DOWNLOADERS.keys()) + ["all"],
        help="Which dataset to download (default: all)."
    )
    p.add_argument(
        "--kaggle_json", type=str, default=None,
        help="Path to kaggle.json (needed in Colab / non-standard envs)."
    )
    args = p.parse_args()

    if args.kaggle_json:
        setup_kaggle(args.kaggle_json)

    if args.dataset == "all":
        download_all(args.data_root)
    else:
        _DOWNLOADERS[args.dataset](args.data_root)
