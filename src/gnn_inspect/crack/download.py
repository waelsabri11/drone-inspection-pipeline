"""Fetch the METU concrete-crack dataset (Özgenel & Gönenç Sorguç, ISARC 2018), CC BY 4.0.

Source: https://data.mendeley.com/datasets/5y9wdsg2zt/2  (no login). The archive is verified against the
SHA-256 published by Mendeley, extracted with bsdtar into its own folder and only ever opened as images.
"""

import hashlib
import subprocess
import urllib.request

from .data import RAW

URL = "https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded"
SHA256 = "08d7dc505a4f5a0330cee2fa2a1ae4b5b4f98bcffd0549b774caf7959bb1f02f"


DEEPCRACK = ("https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip",
             "ec3fc2bee3b71c2cc3c74739cbc51c97b77f78193d08ce3dda0e16d7d41bf585")
CF_API = "https://api.github.com/repos/cuilimeng/CrackForest-dataset/contents/"


def _get(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())


def _cross_datasets():
    """DeepCrack (SHA-256 checked) and CrackForest (118 images + .mat masks) for the cross-dataset test."""
    import json

    raw = RAW.parent
    dc = raw / "deepcrack"
    if not (dc / "train_img").exists():
        dc.mkdir(parents=True, exist_ok=True)
        _get(DEEPCRACK[0], dc / "DeepCrack.zip")
        assert hashlib.sha256((dc / "DeepCrack.zip").read_bytes()).hexdigest() == DEEPCRACK[1], "checksum mismatch"
        subprocess.run(["bsdtar", "-xf", "DeepCrack.zip"], cwd=dc, check=True)
    cf = raw / "crackforest"
    if not (cf / "groundTruth").exists():
        for sub in ("image", "groundTruth"):
            (cf / sub).mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(CF_API + sub, headers={"User-Agent": "Mozilla/5.0"})
            for item in json.load(urllib.request.urlopen(req)):
                if item["name"].endswith((".jpg", ".mat")):
                    _get(item["download_url"], cf / sub / item["name"])
    print("cross-dataset data ready in", raw)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    rar = RAW / "crack.rar"
    if not (RAW / "Positive").exists():
        if not rar.exists():
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(rar, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        assert hashlib.sha256(rar.read_bytes()).hexdigest() == SHA256, "checksum mismatch"
        subprocess.run(["bsdtar", "-xf", rar.name], cwd=RAW, check=True)
    print("METU ready in", RAW)
    _cross_datasets()


if __name__ == "__main__":
    main()
