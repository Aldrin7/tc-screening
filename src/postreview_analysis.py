"""Post-review supplementary analyses (normalization ablation and related checks).

Parts (all LightGBM = primary model unless stated):
  1. family_census    : family sieve counts + per-fold train/test sizes (family_blocked)
  2. weighted_means   : test-size-weighted regime means from released main_results
  3. mondrian         : group-partitioned (Mondrian) split-conformal recalibration
                        - family_blocked: bins of PREDICTED Tc (knots 5/20/50 K)
                        - element_leaveout: bins of element-set size (1-3/4/5/>=6)
                        Note: stratification BY family is impossible in
                        leave-one-family-out (the held-out family has no
                        calibration points); binned groupings are deployable.
  4. jaccard          : distance-to-training diagnostic (min Jaccard distance
                        between element sets) vs |error| and coverage, from
                        RELEASED out-of-fold predictions (no retraining).

Reuses the exact split + calibration-mask logic of the main run
(splits.iter_regime; rng(seed+fold) 25 % calibration mask) so the supplementary
fits are commensurable with the released 630-fit grid.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
import importlib.metadata as md

import numpy as np
import pandas as pd

from features import load_feature_bundle
from models import SCOPE, make_model
from proj_paths import RESULTS_LOGS, RESULTS_TABLES
from splits import iter_regime

LOG = logging.getLogger("postreview")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEEDS = [SCOPE["random_seed_base"] + s for s in range(SCOPE["seeds_folds"]["seeds"])]
ALPHA = 0.10
TC_BIN_KNOTS = (5.0, 20.0, 50.0)          # fixed, declared
ELEM_BIN_EDGES = (3, 4, 5)                # bins: 1-3, 4, 5, >=6 (element count)
MIN_BIN_CAL = 50                          # below this, Mondrian falls back to global q


def _q(resid_abs: np.ndarray) -> float:
    n = len(resid_abs)
    return float(np.quantile(resid_abs, min(1.0, (1 - ALPHA) * (1 + 1 / n))))


def _bin_labels_tc(yhat: np.ndarray) -> np.ndarray:
    b = np.digitize(yhat, TC_BIN_KNOTS)
    return np.array(["<5", "5-20", "20-50", ">=50"])[b]


def _bin_labels_nel(n_el: np.ndarray) -> np.ndarray:
    out = np.full(len(n_el), ">=6", dtype=object)
    out[n_el <= ELEM_BIN_EDGES[0]] = "1-3"
    out[n_el == ELEM_BIN_EDGES[1]] = "4"
    out[n_el == ELEM_BIN_EDGES[2]] = "5"
    return out


# ---------------------------------------------------------------- part 1
def family_census(bundle) -> pd.DataFrame:
    fams = bundle.families.to_numpy()
    rows = []
    for fold, held in enumerate(sorted(set(fams))):
        te = fams == held
        tr = ~te
        y = bundle.y.to_numpy()
        rows.append({
            "fold": fold, "family": held, "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_Tc_mean_K": float(y[te].mean()), "test_Tc_median_K": float(np.median(y[te])),
        })
    df = pd.DataFrame(rows)
    # element-presence census for the leaveout description
    counts = {c: int(sum(c in s for s in bundle.elem_psets)) for c in bundle.elem_cols}
    eligible = sorted(c for c, k in counts.items() if k >= 100)
    df.attrs["n_eligible_elements"] = len(eligible)
    df.to_csv(RESULTS_TABLES / "postreview_family_census.csv", index=False)
    LOG.info("family census: %s | eligible elements (>=100 records): %d",
             df[["family", "n_train", "n_test"]].to_dict("records"), len(eligible))
    return df


# ---------------------------------------------------------------- part 2
def weighted_means() -> pd.DataFrame:
    r = pd.read_csv(RESULTS_TABLES / "main_results.csv")
    out = []
    for (regime, model), g in r.groupby(["regime", "model"]):
        w = g["n_test"].to_numpy()
        out.append({
            "regime": regime, "model": model, "n_fits": len(g),
            "rmse_unweighted": float(np.average(g["rmse_K"])),
            "rmse_weighted_by_ntest": float(np.average(g["rmse_K"], weights=w)),
            "cov_unweighted": float(np.average(g["coverage_90"])),
            "cov_weighted_by_ntest": float(np.average(g["coverage_90"], weights=w)),
        })
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_TABLES / "postreview_weighted_means.csv", index=False)
    return df


# ---------------------------------------------------------------- part 3
def mondrian(bundle) -> pd.DataFrame:
    """Group-partitioned (Mondrian) split-conformal for lightgbm.

    Per (seed, fold): reproduce the exact split + 25 % calibration mask, fit
    lightgbm, compute global q (baseline sanity check) AND per-group quantiles
    from calibration residuals; evaluate coverage on the test fold per group
    (grouping by PREDICTED value at inference, i.e. deployable).
    """
    y = bundle.y.to_numpy()
    nel = np.array([len(s) for s in bundle.elem_psets])
    rows_out = []
    # resume: skip (regime, seed, fold) blocks already pooled in a previous run
    out_path = RESULTS_TABLES / "postreview_mondrian.csv"
    done_blocks: set[tuple] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_blocks = {(r["regime"], int(r["seed"]), int(r["fold"]))
                       for r in prev.to_dict("records") if r["bin"] == "__pooled__"}
        rows_out = prev.to_dict("records")
        LOG.info("mondrian resume: %d blocks already done", len(done_blocks))

    for regime, bin_fn in (("family_blocked", "tc"), ("element_leaveout", "nel")):
        for seed, fold, (tr, te) in iter_regime(regime, bundle, SEEDS, 5):
            if (regime, int(seed), int(fold)) in done_blocks:
                continue
            t0 = time.perf_counter()
            mask = np.random.default_rng(seed + fold).random(len(tr)) < 0.25
            cal_idx, fit_idx = tr[mask], tr[~mask]
            model = make_model("lightgbm", seed)
            model.fit(bundle.X_both.iloc[fit_idx], y[fit_idx])
            pred_cal = np.asarray(model.predict(bundle.X_both.iloc[cal_idx]), float)
            pred_te = np.asarray(model.predict(bundle.X_both.iloc[te]), float)
            yte = y[te]
            resid_cal = np.abs(y[cal_idx] - pred_cal)
            q_global = _q(resid_cal)

            if bin_fn == "tc":
                cal_bins = _bin_labels_tc(pred_cal)
                te_bins = _bin_labels_tc(pred_te)
                grouping = "pred_tc_bin"
            else:
                cal_bins = _bin_labels_nel(nel[cal_idx])
                te_bins = _bin_labels_nel(nel[te])
                grouping = "elem_count_bin"

            # baseline: global interval
            cov_base = np.mean(np.abs(yte - pred_te) <= q_global)
            # per-bin conditional coverage for BASELINE (diagnostic)
            # Mondrian: per-bin quantiles with fallback to global q
            te_cov = np.abs(yte - pred_te)
            mon_cov = np.zeros(len(te), bool)
            mon_w = np.zeros(len(te))
            n_fallback = 0
            for b in sorted(set(cal_bins) | set(te_bins)):
                tm = te_bins == b
                cm = cal_bins == b
                if tm.sum() == 0:
                    continue
                if cm.sum() >= MIN_BIN_CAL:
                    qb = _q(resid_cal[cm])
                else:
                    qb = q_global
                    n_fallback += 1
                mon_cov[tm] = te_cov[tm] <= qb
                mon_w[tm] = 2 * qb
                rows_out.append({
                    "regime": regime, "grouping": grouping, "seed": seed, "fold": fold,
                    "bin": b, "n_cal_bin": int(cm.sum()), "n_test_bin": int(tm.sum()),
                    "fallback": bool(cm.sum() < MIN_BIN_CAL),
                    "cov_baseline_bin": float(np.mean(te_cov[tm] <= q_global)),
                    "cov_mondrian_bin": float(np.mean(mon_cov[tm])),
                    "width_mondrian_bin": float(2 * qb),
                    "bin_width_baseline_global": float(2 * q_global),
                    "bin_mean_abserr": float(np.mean(te_cov[tm])),
                })
            rows_out.append({
                "regime": regime, "grouping": grouping, "seed": seed, "fold": fold,
                "bin": "__pooled__", "n_cal_bin": len(cal_idx), "n_test_bin": len(te),
                "fallback": None, "n_bins_fallback": n_fallback,
                "cov_baseline_bin": float(cov_base),
                "cov_mondrian_bin": float(np.mean(mon_cov)),
                "width_mondrian_bin": float(np.mean(mon_w)),
                "bin_width_baseline_global": float(2 * q_global),
                "bin_mean_abserr": float(np.mean(te_cov)),
            })
            LOG.info("mondrian %s s%d f%d done (%.1fs) pooled base=%.3f mon=%.3f",
                     regime, seed, fold, time.perf_counter() - t0, cov_base, np.mean(mon_cov))
            pd.DataFrame(rows_out).to_csv(
                RESULTS_TABLES / "postreview_mondrian.csv", index=False)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------- part 4
def jaccard_diagnostic(bundle) -> pd.DataFrame:
    """Min Jaccard distance (element sets) to the training fold vs |error| and
    coverage, using RELEASED OOF predictions (lightgbm, seed 20260101)."""
    seed = SEEDS[0]
    preds = pd.read_parquet(RESULTS_TABLES / "main_predictions.parquet")
    results = pd.read_csv(RESULTS_TABLES / "main_results.csv")
    E = (bundle.X_elem.to_numpy() > 0)
    n_el = E.sum(axis=1)
    Ef = E.astype(np.float32)
    out = []
    for regime in ("family_blocked", "element_leaveout"):
        for fold in sorted(preds.loc[preds.regime == regime, "fold"].unique()):
            p = preds[(preds.regime == regime) & (preds.model == "lightgbm") &
                      (preds.seed == seed) & (preds.fold == fold)]
            if p.empty:
                continue
            q = results[(results.regime == regime) & (results.model == "lightgbm") &
                        (results.seed == seed) & (results.fold == fold)]["conformal_q"].iloc[0]
            tr_mask = np.ones(len(E), bool)
            tr_mask[p["row"].to_numpy()] = False
            tr_idx = np.where(tr_mask)[0]
            te_idx = p["row"].to_numpy()
            dmin = np.empty(len(te_idx), dtype=np.float32)
            CH = 2048
            for s in range(0, len(te_idx), CH):
                chunk = Ef[te_idx[s:s + CH]]
                inter = chunk @ Ef[tr_idx].T
                union = (n_el[te_idx[s:s + CH], None] + n_el[tr_idx][None, :] - inter)
                dmin[s:s + CH] = 1.0 - (inter / np.maximum(union, 1)).max(axis=1)
            abs_err = np.abs(p["y_true"].to_numpy() - p["y_pred"].to_numpy())
            covered = abs_err <= q
            ds = pd.Series(dmin)
            dec = pd.qcut(ds, 10, duplicates="drop")
            for interval, idx in ds.groupby(dec, observed=True).groups.items():
                idx = np.asarray(idx)
                out.append({
                    "regime": regime, "fold": int(fold),
                    "jaccard_decile": str(interval), "n": int(len(idx)),
                    "dmin_mean": float(np.mean(dmin[idx])),
                    "abs_err_mean": float(np.mean(abs_err[idx])),
                    "rmse": float(np.sqrt(np.mean(abs_err[idx] ** 2))),
                    "coverage": float(np.mean(covered[idx])),
                    "y_true_mean": float(np.mean(p["y_true"].to_numpy()[idx])),
                })
            rho = pd.Series(dmin).corr(pd.Series(abs_err), method="spearman")
            out.append({"regime": regime, "fold": int(fold), "jaccard_decile": "__all__",
                        "n": len(te_idx), "dmin_mean": float(np.mean(dmin)),
                        "abs_err_mean": float(np.mean(abs_err)),
                        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
                        "coverage": float(np.mean(covered)),
                        "spearman_dmin_abserr": float(rho),
                        "y_true_mean": float(np.mean(p["y_true"].to_numpy()))})
            LOG.info("jaccard %s f%d: spearman(d,|err|)=%.3f pooled_cov=%.3f",
                     regime, fold, rho, float(np.mean(covered)))
            pd.DataFrame(out).to_csv(
                RESULTS_TABLES / "postreview_jaccard.csv", index=False)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- part 5
def norm_ablation(bundle) -> pd.DataFrame:
    """Post-review ablation (reviewer Q2/Q3, both rounds): element-fraction block
    normalized to unit row sum vs the raw (source-distributed) fractions.
    LightGBM (primary model) only; iid_random + formula_grouped; 5 seeds x 5 folds;
    full-fold training, exactly mirroring extras.run_ablations (no calibration mask)
    so the released elem_only numbers and this run share one protocol. Both variants
    are (re)run here in a single environment, giving a paired, cross-kernel-free
    comparison."""
    from train_eval import metrics
    y = bundle.y.to_numpy()
    row_sums = bundle.X_elem.sum(axis=1)
    assert (row_sums > 0).all(), "zero element-fraction row sum found"
    Xn = bundle.X_elem.div(row_sums, axis=0)
    sets = {"elem_raw": bundle.X_elem, "elem_unitnorm": Xn}
    out_path = RESULTS_TABLES / "postreview_norm_ablation.csv"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = {(r["set"], r["regime"], int(r["seed"]), int(r["fold"])) for r in rows}
        LOG.info("norm ablation resume: %d fits done", len(done))
    for regime in ("iid_random", "formula_grouped"):
        for sname, X in sets.items():
            for seed, fold, (tr, te) in iter_regime(regime, bundle, SEEDS, 5):
                key = (sname, regime, int(seed), int(fold))
                if key in done:
                    continue
                model = make_model("lightgbm", seed)
                model.fit(X.iloc[tr], y[tr])
                m = metrics(y[te], model.predict(X.iloc[te]))
                rows.append({"set": sname, "regime": regime, "model": "lightgbm",
                             "seed": int(seed), "fold": int(fold),
                             "n_features": X.shape[1], **m})
                pd.DataFrame(rows).to_csv(out_path, index=False)  # step saving
                done.add(key)
    df = pd.DataFrame(rows)
    LOG.info("norm ablation summary:\n%s",
             df.groupby(["set", "regime"])["rmse_K"].agg(["mean", "std"]).round(3).to_string())
    return df


def main() -> None:
    t0 = time.perf_counter()
    bundle = load_feature_bundle()
    manifest = {
        "purpose": "post-review supplementary analyses",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {p: md.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "lightgbm")},
        "n_fits_planned": 55, "model": "lightgbm",
        "norm_ablation_fits": 100,
        "tc_bin_knots_K": list(TC_BIN_KNOTS),
        "elem_count_bin_edges": list(ELEM_BIN_EDGES),
        "min_bin_calibration_rows": MIN_BIN_CAL,
        "note": ("split + calibration-mask logic identical to main run "
                 "(splits.iter_regime, rng(seed+fold) < 0.25); Mondrian grouping "
                 "by family is impossible in leave-one-family-out (no calibration "
                 "points for the held-out family), hence deployable binned groupings"),
    }
    (RESULTS_LOGS / "postreview_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    family_census(bundle)
    wm = weighted_means()
    LOG.info("weighted means (lightgbm):\n%s",
             wm[wm.model == "lightgbm"].to_string(index=False))
    mondrian(bundle)
    jaccard_diagnostic(bundle)
    norm_ablation(bundle)
    manifest["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (RESULTS_LOGS / "postreview_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("post-review analysis complete in %.1f min", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
