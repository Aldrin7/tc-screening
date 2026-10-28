"""Package the Tc-screening pipeline for Kaggle and launch concurrent CPU kernels.

Creates:
  C:\\Users\\aldri\\tc_kaggle\\code_dataset          -> Kaggle dataset tc-screening-code
  C:\\Users\\aldri\\tc_kaggle\\kernel_main_a         -> kernel tc-screening-main-a
      gate-b + main CV on iid_random + formula_grouped -> main_results_a
  C:\\Users\\aldri\\tc_kaggle\\kernel_main_b         -> kernel tc-screening-main-b
      main CV on family_blocked + element_leaveout     -> main_results_b
  C:\\Users\\aldri\\tc_kaggle\\kernel_extras_1       -> kernel tc-screening-extras-1
      extras part 1: feature-set ablations
  C:\\Users\\aldri\\tc_kaggle\\kernel_extras_2       -> kernel tc-screening-extras-2
      extras part 2: top-k removal, SHAP stability, probes, residuals, dedup

Split rationale: measured local throughput ~7 min per (regime,seed,fold) block of
6 models and ~2.2 min per LightGBM fit on 4 threads -> each kernel stays well under
Kaggle's 12 h CPU session cap. The pipeline checkpoints after every fit and the
merged results are assembled locally after all kernels complete.

No credentials are read, printed, or written by this script; the Kaggle CLI uses
~/.kaggle/access_token directly.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import importlib.metadata as md

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
STAGE = Path.home() / "tc_kaggle"

USERNAME = "aldrinmanon"
CODE_SLUG = "tc-screening-code"
DATA_SLUG = "tunguz/superconductivty-data-data-set"

KERNEL_TEMPLATE = '''"""Tc-screening kernel (mode: __MODE__).

__DESCRIPTION__

All outputs are written under /kaggle/working/TcScreening (persisted as kernel
output). The pipeline checkpoints after every model fit and logs per-fit metrics;
tqdm progress is visible in the notebook log while it runs.
"""
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = WORK / "TcScreening"
CODE_IN = Path("/kaggle/input/__CODE_SLUG__")
DATA_IN = Path("/kaggle/input/superconductivty-data-data-set")
MODE = "__MODE__"


def find_file(root: Path, name: str):
    """Locate a file anywhere under the input mount (layout-agnostic)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if name in filenames:
            return Path(dirpath) / name
    return None


def find_dir(root: Path, name: str):
    """Locate a directory anywhere under the input mount (layout-agnostic)."""
    for dirpath, dirnames, _filenames in os.walk(root):
        if name in dirnames:
            return Path(dirpath) / name
    return None


def prepare():
    import os
    PROJ.mkdir(parents=True, exist_ok=True)
    root = Path("/kaggle/input")
    if not root.exists():
        raise FileNotFoundError("/kaggle/input does not exist")
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    print("INPUT FILES (%d): %s" % (len(files), files[:80]), flush=True)
    zpath = find_file(root, "code_bundle.zip")
    if zpath is not None:
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(PROJ)
    else:
        bundle_dir = find_dir(root, "code_bundle")
        if bundle_dir is None:
            raise FileNotFoundError(
                "neither code_bundle.zip nor code_bundle/ found under /kaggle/input")
        shutil.copytree(bundle_dir, PROJ, dirs_exist_ok=True)
    tpath = find_file(root, "train.csv")
    upath = find_file(root, "unique_m.csv")
    if tpath is None or upath is None:
        raise FileNotFoundError("SuperCon CSVs not found under /kaggle/input")
    dk = PROJ / "data" / "raw" / "supercon_kaggle"
    dk.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tpath, dk / "train.csv")
    shutil.copy2(upath, dk / "unique_m.csv")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "catboost", "shap", "tabulate", "pyarrow"],
                   check=False, capture_output=True)


def run(args):
    cmd = [sys.executable, "run_pipeline.py"] + args
    print("RUN:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJ)
    print("DONE exit=%d elapsed=%.0fs" % (proc.returncode, time.time() - t0), flush=True)
    return proc.returncode


STAGES = {
    "main-a": [
        ["--stage", "gate-b"],
        ["--stage", "main", "--regimes", "iid_random,formula_grouped",
         "--out-name", "main_results_a"],
    ],
    "main-b": [
        ["--stage", "main", "--regimes", "family_blocked,element_leaveout",
         "--out-name", "main_results_b"],
    ],
    "extras-1": [
        ["--stage", "extras", "--extras-part", "1"],
    ],
    "extras-2": [
        ["--stage", "extras", "--extras-part", "2"],
    ],
}


def main():
    print("mode=%s python=%s" % (MODE, sys.version), flush=True)
    prepare()
    rc = 0
    for args in STAGES[MODE]:
        rc = run(args)
        if rc != 0:
            traceback.print_exc()
            break
    tables = PROJ / "results" / "tables"
    if tables.exists():
        for p in sorted(tables.glob("*")):
            print("OUTPUT: %s %d" % (p.name, p.stat().st_size), flush=True)
    sys.exit(rc)


main()
'''

KERNELS = [
    ("main-a", "tc-screening-main-a",
     "Gate B checks + main CV (iid_random, formula_grouped) x 6 models x 5 seeds."),
    ("main-b", "tc-screening-main-b",
     "Main CV (family_blocked, element_leaveout) x 6 models x 5 seeds."),
    ("extras-1", "tc-screening-extras-1",
     "Feature-set ablations (stat 81 / elem 86 / both 167) x {lightgbm, ridge} x 2 regimes."),
    ("extras-2", "tc-screening-extras-2",
     "Top-k SHAP removal, SHAP stability, scenario probes, residuals, dedup sensitivity."),
]


def refresh_requirements() -> None:
    pkgs = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost",
            "shap", "scipy", "matplotlib", "seaborn", "pyarrow", "joblib", "tqdm",
            "requests", "kaggle", "pyyaml", "tabulate"]
    lines = [f"{p}=={md.version(p)}" for p in pkgs]
    (PROJECT_ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("requirements.txt refreshed (%d pins)" % len(lines))


def stage_code_dataset() -> Path:
    d = STAGE / "code_dataset"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "run_pipeline.py", d / "run_pipeline.py")
    shutil.copy2(PROJECT_ROOT / "requirements.txt", d / "requirements.txt")
    shutil.copy2(PROJECT_ROOT / "README.md", d / "README.md")
    # the Kaggle CLI does not upload subdirectories reliably; ship the whole
    # project tree as one zip that the kernel extracts into /kaggle/working
    import zipfile
    zip_path = d / "code_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in ("configs", "src"):
            for p in (PROJECT_ROOT / base).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    zf.write(p, Path(base) / p.relative_to(PROJECT_ROOT / base))
        for name in ("run_pipeline.py", "requirements.txt", "README.md"):
            zf.write(PROJECT_ROOT / name, name)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": CODE_SLUG,
        "id": f"{USERNAME}/{CODE_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")
    return d


def stage_kernel(mode: str, slug: str, description: str) -> Path:
    d = STAGE / f"kernel_{mode.replace('-', '_')}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    script = (KERNEL_TEMPLATE
              .replace("__MODE__", mode)
              .replace("__DESCRIPTION__", description)
              .replace("__CODE_SLUG__", CODE_SLUG))
    (d / "tc_screening_kernel.py").write_text(script, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USERNAME}/{slug}",
        "title": slug,
        "code_file": "tc_screening_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{USERNAME}/{CODE_SLUG}", DATA_SLUG],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")
    return d


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    print("+ kaggle", " ".join(args), flush=True)
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True, **kw)


def main() -> None:
    refresh_requirements()
    code_dir = stage_code_dataset()
    kernel_dirs = [stage_kernel(*k) for k in KERNELS]

    # 1. create the code dataset (or push a new version).
    # NOTE: kaggle CLI 2.2.4 prints errors but still exits 0 -> check the text.
    r = sh(["datasets", "create", "-p", str(code_dir)])
    combined = (r.stdout + r.stderr).lower()
    if "already" in combined and "in use" in combined:
        print("dataset exists -> pushing a new version instead")
        r = sh(["datasets", "version", "-p", str(code_dir),
                "-m", "bundle zip: full project tree (src/configs)"])
        combined = (r.stdout + r.stderr).lower()
    if "error" in combined and "upload" not in combined:
        print("DATASET PUSH FAILED:", r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)
    # 1b. wait until the dataset is processed (kernels need it mounted)
    ready = False
    for _ in range(20):
        time.sleep(30)
        s = sh(["datasets", "status", f"{USERNAME}/{CODE_SLUG}"])
        state = (s.stdout + s.stderr).strip().lower()
        print("dataset status:", state[-80:], flush=True)
        if "ready" in state:
            ready = True
            break
    if not ready:
        print("dataset not ready after 10 min; aborting kernel push")
        sys.exit(1)

    # 2. push the four kernels
    failed = []
    for (mode, slug, _), kd in zip(KERNELS, kernel_dirs):
        r = sh(["kernels", "push", "-p", str(kd)])
        if r.returncode != 0:
            print(f"PUSH FAILED for {slug}:", r.stdout[-800:], r.stderr[-800:])
            failed.append(slug)
        else:
            print(f"pushed {slug}: {r.stdout.strip()[-200:]}")
    if failed:
        sys.exit(1)
    print("ALL 4 KERNELS PUSHED AND QUEUED")


if __name__ == "__main__":
    main()
