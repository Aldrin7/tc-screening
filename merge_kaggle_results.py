"""Merge the two partitioned Kaggle main-CV outputs into the canonical results.

main-a: iid_random + formula_grouped  (out_name main_results_a)
main-b: family_blocked + element_leaveout  (out_name main_results_b)

Asserts the declared grid: 630 fits total
  iid_random       25 (seed,fold) blocks x 6 models = 150 rows
  formula_grouped  25 x 6 = 150
  family_blocked   30 x 6 = 180
  element_leaveout 25 x 6 = 150
Also merges the per-row out-of-fold predictions into one parquet.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
KA = Path.home() / "tc_kaggle"
KEY = ["regime", "model", "seed", "fold"]

EXPECTED = {
    "iid_random": 150,
    "formula_grouped": 150,
    "family_blocked": 180,
    "element_leaveout": 150,
}


def main() -> None:
    tables = ROOT / "results" / "tables"
    a = pd.read_parquet(KA / "out_main_a" / "TcScreening" / "results" / "tables"
                        / "main_results_a.parquet")
    b = pd.read_parquet(KA / "out_main_b" / "TcScreening" / "results" / "tables"
                        / "main_results_b.parquet")
    df = pd.concat([a, b], ignore_index=True)
    dup = df.duplicated(subset=KEY).sum()
    assert dup == 0, f"{dup} duplicate (regime,model,seed,fold) keys after merge"
    counts = df.groupby("regime").size().to_dict()
    print("fits per regime:", counts)
    assert counts == EXPECTED, f"regime counts {counts} != expected {EXPECTED}"
    assert len(df) == 630, f"total fits {len(df)} != 630"
    df = df.sort_values(KEY).reset_index(drop=True)
    df.to_parquet(tables / "main_results.parquet")
    df.to_csv(tables / "main_results.csv", index=False)
    print(f"main_results written: {len(df)} fits")

    # per-row out-of-fold predictions
    pa = pd.read_csv(KA / "out_main_a" / "TcScreening" / "results" / "tables"
                     / "main_results_a_predictions.csv")
    pb = pd.read_csv(KA / "out_main_b" / "TcScreening" / "results" / "tables"
                     / "main_results_b_predictions.csv")
    preds = pd.concat([pa, pb], ignore_index=True)
    dupp = preds.duplicated(subset=KEY + ["row"]).sum()
    assert dupp == 0, f"{dupp} duplicate prediction rows after merge"
    preds = preds.sort_values(KEY + ["row"]).reset_index(drop=True)
    preds.to_parquet(tables / "main_predictions.parquet")
    print(f"main_predictions written: {len(preds)} rows")

    # fit-time summary + manifests for the Gate D report
    t = df.groupby("model")["fit_seconds"].agg(["mean", "max", "sum"]).round(2)
    print("\nfit_seconds by model:\n", t)
    for src, dst in ((KA / "out_main_a" / "TcScreening" / "results" / "logs" / "run_manifest.json",
                      ROOT / "results" / "logs" / "kaggle_run_manifest_a.json"),
                     (KA / "out_main_b" / "TcScreening" / "results" / "logs" / "run_manifest.json",
                      ROOT / "results" / "logs" / "kaggle_run_manifest_b.json")):
        if src.exists():
            shutil.copy2(src, dst)
            print("copied manifest ->", dst.name)


if __name__ == "__main__":
    main()
