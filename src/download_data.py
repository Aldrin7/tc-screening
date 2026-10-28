"""Gate A step 1: download the SuperCon-derived dataset from two independent mirrors.

Mirror 1 (Kaggle): tunguz/superconductivty-data-data-set (copy of the Hamidieh 2018 files).
Mirror 2 (UCI):    UCI Machine Learning Repository dataset 464 "Superconductivty Data"
                   (canonical DOI: 10.24432/C53P47; contributor: Kam Hamidieh).

Both are derived from the NIMS SuperCon database (http://supercon.nims.go.jp/index_en.html)
via Hamidieh 2018, Comput. Mater. Sci. 154 (2018) 346-354,
DOI 10.1016/j.commatsci.2018.07.052. This script records hashes + access dates in
data/curation/data_manifest.json. NOTHING is assumed; everything is asserted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

from proj_paths import DATA_KAGGLE, DATA_UCI, MANIFEST_PATH

LOG = logging.getLogger("download_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KAGGLE_SLUG = "tunguz/superconductivty-data-data-set"
UCI_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
UCI_DOI = "10.24432/C53P47"
KAGGLE_DOI_ALTERNATES = "none listed (Kaggle dataset page; verify licence at Gate A)"
ACCESS_DATE = date.today().isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_kaggle(target: Path) -> list[Path]:
    """Download via the official kaggle CLI (reads ~/.kaggle/access_token)."""
    target.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", KAGGLE_SLUG, "-p", str(target), "--unzip"]
    LOG.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download failed: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    files = sorted(target.glob("*.csv"))
    assert files, "no CSV files found after Kaggle download"
    LOG.info("kaggle files: %s", [f.name for f in files])
    return files


def download_uci(target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "superconductivty_uci.zip"
    if not (target / "train.csv").exists():
        LOG.info("GET %s", UCI_URL)
        resp = requests.get(UCI_URL, timeout=300, headers={"User-Agent": "tc-screening-research/0.1"})
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        LOG.info("extracted: %s", [p.name for p in target.iterdir()])
    files = sorted(target.glob("*.csv"))
    assert files, "no CSV files found after UCI download"
    return files


def main() -> None:
    kaggle_files = download_kaggle(DATA_KAGGLE)
    uci_files = download_uci(DATA_UCI)

    manifest = {
        "generated_on": ACCESS_DATE,
        "project": "Tc screening (Materials Futures paper)",
        "canonical_origin": {
            "database": "NIMS SuperCon",
            "url": "http://supercon.nims.go.jp/index_en.html",
            "derived_by": "Hamidieh, K. A data-driven statistical model for predicting the "
                          "critical temperature of a superconductor. Comput. Mater. Sci. 154, 346-354 (2018). "
                          "DOI 10.1016/j.commatsci.2018.07.052",
        },
        "mirrors": [],
    }
    for label, folder, files in (("kaggle", DATA_KAGGLE, kaggle_files),
                                 ("uci", DATA_UCI, uci_files)):
        manifest["mirrors"].append({
            "source": label,
            "url": UCI_URL if label == "uci" else f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
            "doi": UCI_DOI if label == "uci" else KAGGLE_DOI_ALTERNATES,
            "access_date": ACCESS_DATE,
            "files": {f.name: {"size_bytes": f.stat().st_size,
                               "sha256": sha256(f)} for f in files},
        })
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("manifest written to %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
