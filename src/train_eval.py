"""Training + evaluation loops (Gates C/D): CV across regimes, metrics,
split-conformal intervals, run manifest. Results saved as parquet + summaries.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import importlib.metadata as md

from features import load_feature_bundle, TARGET
from models import SCOPE, make_model
from proj_paths import CONFIG_DIR, RESULTS_LOGS, RESULTS_TABLES
from splits import iter_regime, fold_counts

LOG = logging.getLogger("train_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

METRICS = ("rmse_K", "mae_K", "r2")


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    return {"rmse_K": float(np.sqrt(np.mean(err ** 2))),
            "mae_K": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))}


def conformal(y_cal_pred: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal absolute-residual quantile (nominal coverage 1 - alpha).

    Returns the half-width q; intervals on a test fold are [pred - q, pred + q].
    Measured coverage is computed separately on the test fold (validity check).
    """
    q = np.quantile(np.abs(y_cal - y_cal_pred), min(1.0, (1 - alpha) * (1 + 1 / len(y_cal))))
    return float(q)


def run_main_cv(regimes: list[str], models: list[str], out_name: str = "main_results",
                seeds_run: list[int] | None = None, folds_run: int | None = None,
                resume: bool = True) -> pd.DataFrame:
    """Full CV with per-fit checkpointing (step saving) and resume-on-restart.

    After EVERY completed fit the results parquet is rewritten, so an interrupted
    run resumes from the last completed (regime, model, seed, fold) key instead of
    starting over. Set resume=False to force a clean run.
    """
    from tqdm import tqdm

    bundle = load_feature_bundle()
    y = bundle.y.to_numpy()
    cfg_seeds = SCOPE["seeds_folds"]["seeds"]
    folds = folds_run or SCOPE["seeds_folds"]["folds_per_seed"]
    seeds = seeds_run or [SCOPE["random_seed_base"] + s for s in range(cfg_seeds)]

    # run manifest
    manifest = {
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: md.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap")},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope_sha256": _sha256(CONFIG_DIR / "scope.yaml"),
    }
    (RESULTS_LOGS / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    out_parquet = RESULTS_TABLES / f"{out_name}.parquet"
    out_csv = RESULTS_TABLES / f"{out_name}.csv"
    out_preds = RESULTS_TABLES / f"{out_name}_predictions.csv"
    rows_by_key: dict[tuple, dict] = {}
    if resume and out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        for r in prev.to_dict("records"):
            rows_by_key[(r["regime"], r["model"], int(r["seed"]), int(r["fold"]))] = r
        LOG.info("resume: %d result rows loaded", len(rows_by_key))
    # a fit counts as done only if BOTH its result row and its per-row predictions
    # exist (older partial runs without a predictions file get re-run; stale rows
    # are superseded by key)
    preds_keys: set[tuple] = set()
    if resume and out_preds.exists():
        prev_p = pd.read_csv(out_preds, usecols=["regime", "model", "seed", "fold"])
        for r in prev_p.to_dict("records"):
            preds_keys.add((r["regime"], r["model"], int(r["seed"]), int(r["fold"])))
        LOG.info("resume: %d prediction blocks loaded", len(preds_keys))
    done = set(rows_by_key) & preds_keys if resume else set()
    LOG.info("resume: %d fully-done (regime,model,seed,fold) keys will be skipped", len(done))

    # scope declaration: log fold sanity per regime
    for regime in regimes:
        LOG.info("fold census %s: %s", regime, fold_counts(regime, bundle, seeds))

    # plan all fits up front so the progress bar reflects the entire run
    plan = [(regime, seed, fold, tr, te)
            for regime in regimes
            for seed, fold, (tr, te) in iter_regime(regime, bundle, seeds, folds)]
    total_fits = len(plan) * len(models)
    LOG.info("planned fits: %d (%d (regime,seed,fold) blocks x %d models); %d already done",
             total_fits, len(plan), len(models), len(done))
    pbar = tqdm(total=total_fits, desc="CV fits", unit="fit", dynamic_ncols=True)

    for regime, seed, fold, tr_idx, te_idx in plan:
        yte = y[te_idx]
        n_tr, n_te = len(tr_idx), len(te_idx)
        split_mask = np.random.default_rng(seed + fold).random(n_tr) < 0.25  # calibration 25%
        cal_idx, fit_idx = tr_idx[split_mask], tr_idx[~split_mask]
        for model_name in models:
            pbar.update(1)
            key = (regime, model_name, int(seed), int(fold))
            if key in done:
                continue
            t0 = time.perf_counter()
            model = make_model(model_name, seed)
            model.fit(bundle.X_both.iloc[fit_idx], y[fit_idx])
            fit_s = time.perf_counter() - t0

            yhat_te = np.asarray(model.predict(bundle.X_both.iloc[te_idx]), dtype=float)
            m = metrics(yte, yhat_te)
            q = conformal(model.predict(bundle.X_both.iloc[cal_idx]), y[cal_idx])
            lo, hi = yhat_te - q, yhat_te + q
            m.update({"coverage_90": float(np.mean((yte >= lo) & (yte <= hi))),
                      "interval_width_mean": float(np.mean(hi - lo)),
                      "conformal_q": q})
            rows_by_key[key] = {"regime": regime, "model": model_name,
                                "seed": int(seed), "fold": int(fold),
                                "n_train": n_tr, "n_calib": len(cal_idx), "n_test": n_te,
                                "fit_seconds": fit_s, **m}
            # STEP SAVING: checkpoint after every single fit (results + predictions)
            pd.DataFrame(list(rows_by_key.values())).to_parquet(out_parquet)
            header = not out_preds.exists()
            pd.DataFrame([{"regime": regime, "model": model_name,
                           "seed": int(seed), "fold": int(fold),
                           "row": int(i), "y_true": float(yt), "y_pred": float(yp)}
                          for i, yt, yp in zip(te_idx, yte, yhat_te)]
                         ).to_csv(out_preds, mode="a", header=header, index=False,
                                  float_format="%.6f")
            done.add(key)
            LOG.info("%s %s s%d f%d rmse=%.3f mae=%.3f r2=%.3f cov=%.3f (%.1fs)",
                     regime, model_name, seed, fold, m["rmse_K"], m["mae_K"], m["r2"],
                     m["coverage_90"], fit_s)
    pbar.close()
    df = pd.DataFrame(list(rows_by_key.values()))
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    df.to_csv(out_csv, index=False)
    (RESULTS_LOGS / "run_manifest.json").write_text(json.dumps(
        {**manifest, "ended": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    return df


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["regime", "model"])[["rmse_K", "mae_K", "r2", "coverage_90", "interval_width_mean"]]
    summ = g.agg(["mean", "std", "count"]).round(4)
    return summ


def summarize_save(df: pd.DataFrame, out_name: str = "main_results_summary") -> pd.DataFrame:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    summ = summarize(df)
    summ.to_csv(RESULTS_TABLES / f"{out_name}.csv")
    try:
        summ.to_markdown(RESULTS_TABLES / f"{out_name}.md")
    except Exception as exc:  # cosmetic export only; CSV already written
        LOG.warning("markdown summary export skipped: %s", exc)
    return summ


def significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests across (seed, fold) for lightgbm vs each other model, per regime,
    with Holm-Bonferroni correction. Only computed here - never claimed elsewhere."""
    base = "lightgbm"
    out = []
    for regime, gdf in df.groupby("regime"):
        pivot = gdf.pivot_table(index=["seed", "fold"], columns="model", values="rmse_K")
        models = [m for m in pivot.columns if m != base]
        raw_p = []
        for m in models:
            t, p = st.ttest_rel(pivot[base], pivot[m])
            out.append({"regime": regime, "model": m,
                        "mean_diff_rmse_K": float(np.mean(pivot[base] - pivot[m])),
                        "t": float(t), "p_raw": float(p),
                        "n_pairs": int(len(pivot))})
            raw_p.append(float(p))
        # Holm-Bonferroni within regime
        order = np.argsort(raw_p)
        holm = [False] * len(raw_p)
        m_total = len(raw_p)
        for rank, i in enumerate(order):
            holm[i] = raw_p[i] < 0.05 / (m_total - rank)
        for o in out[-len(models):]:
            o["holm_reject_0.05"] = holm.pop(0)
    sig = pd.DataFrame(out)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    sig.to_csv(RESULTS_TABLES / "significance_tests.csv", index=False)
    return sig
