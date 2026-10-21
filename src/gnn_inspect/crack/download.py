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
    print("dataset ready in", RAW)


if __name__ == "__main__":
    main()
