"""Ablations, SHAP explainability, scenario probes, residual diagnostics, dedup
sensitivity (post-main stage). Declared in configs/scope.yaml. All loop-heavy
functions save progress after every fit and resume from saved parquet on rerun.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from features import load_feature_bundle  # noqa: F401  (re-exported for run_pipeline)
from models import SCOPE, make_model
from proj_paths import RESULTS_TABLES, FUTURE_SCOPE_DIR
from splits import iter_regime

LOG = logging.getLogger("extras")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEED_BASE = SCOPE["random_seed_base"]
SEEDS = [SEED_BASE + s for s in range(SCOPE["seeds_folds"]["seeds"])]
PRIMARY = "lightgbm"
BASE_MODEL = "ridge"


def _load_done(path, keyfn) -> tuple[list[dict], set[tuple]]:
    """Resume helper: load existing parquet rows and their keys."""
    if path.exists():
        prev = pd.read_parquet(path)
        rows = prev.to_dict("records")
        done = {keyfn(r) for r in rows}
        LOG.info("resume: %d rows loaded from %s (%d keys done)", len(rows), path.name, len(done))
        return rows, done
    return [], set()


def run_ablations(bundle) -> pd.DataFrame:
    """Feature-set ablation: stat_only (81) vs elem_only (86) vs stat_plus_elem (167),
    lightgbm + ridge, regimes iid_random and formula_grouped (5 seeds x 5 folds)."""
    from train_eval import metrics
    y = bundle.y.to_numpy()
    sets = {"stat_only": bundle.X_stat, "elem_only": bundle.X_elem,
            "stat_plus_elem": bundle.X_both}
    out_path = RESULTS_TABLES / "ablation_feature_sets.parquet"
    rows, done = _load_done(out_path, lambda r: (r["set"], r["regime"], r["model"],
                                                 int(r["seed"]), int(r["fold"])))
    for regime in ("iid_random", "formula_grouped"):
        for sname, X in sets.items():
            for mname in (PRIMARY, BASE_MODEL):
                LOG.info("ablation %s %s %s", regime, sname, mname)
                for seed, fold, (tr, te) in iter_regime(regime, bundle, SEEDS, 5):
                    key = (sname, regime, mname, int(seed), int(fold))
                    if key in done:
                        continue
                    model = make_model(mname, seed)
                    model.fit(X.iloc[tr], y[tr])
                    m = metrics(y[te], model.predict(X.iloc[te]))
                    rows.append({"set": sname, "regime": regime, "model": mname,
                                 "seed": int(seed), "fold": int(fold),
                                 "n_train": len(tr), "n_test": len(te),
                                 "n_features": X.shape[1], **m})
                    pd.DataFrame(rows).to_parquet(out_path)  # step saving
                    done.add(key)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return df


def top_shap_features(bundle, k: int = 15, seed: int = SEED_BASE):
    """Mean |SHAP| ranking on the full-data LightGBM fit; returns top-k feature names.
    Reuses the cached full ranking (future_scope/) when present."""
    cache = FUTURE_SCOPE_DIR / "shap_mean_abs_full_order.csv"
    if cache.exists():
        LOG.info("top_shap_features: reusing cached ranking %s", cache)
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]
        return list(s.sort_values(ascending=False).index[:k])
    import shap
    model = make_model("lightgbm", seed)
    model.fit(bundle.X_both, bundle.y.to_numpy())
    expl = shap.TreeExplainer(model)
    sub = bundle.X_both.sample(n=min(2000, len(bundle.X_both)), random_state=seed)
    sv = expl.shap_values(sub)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    ranked = list(np.array(bundle.X_both.columns)[order][:k])
    # save artifact for reuse (future_scope)
    pd.Series(mean_abs, index=bundle.X_both.columns).sort_values(ascending=False) \
        .to_csv(cache)
    return ranked


def run_topk_removal(bundle) -> pd.DataFrame:
    """Drop top-5 / top-10 SHAP-ranked features (combines explainability + ablation)."""
    from train_eval import metrics
    y = bundle.y.to_numpy()
    topk = top_shap_features(bundle, k=15, seed=SEED_BASE)
    out_path = RESULTS_TABLES / "ablation_topk_removal.parquet"
    rows, done = _load_done(out_path, lambda r: (int(r["k"]), r["regime"],
                                                 int(r["seed"]), int(r["fold"])))
    for regime in ("iid_random", "formula_grouped"):
        for k in (5, 10):
            drop = topk[:k]
            keep = [c for c in bundle.X_both.columns if c not in drop]
            X = bundle.X_both[keep]
            LOG.info("top-removal regime=%s k=%d (dropped: %s...)", regime, k, ",".join(drop[:3]))
            for seed, fold, (tr, te) in iter_regime(regime, bundle, SEEDS, 5):
                key = (k, regime, int(seed), int(fold))
                if key in done:
                    continue
                model = make_model(PRIMARY, seed)
                model.fit(X.iloc[tr], y[tr])
                m = metrics(y[te], model.predict(X.iloc[te]))
                rows.append({"k": k, "regime": regime, "model": PRIMARY,
                             "seed": int(seed), "fold": int(fold),
                             "n_train": len(tr), "n_test": len(te),
                             "n_features": X.shape[1], "dropped": ",".join(drop), **m})
                pd.DataFrame(rows).to_parquet(out_path)  # step saving
                done.add(key)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return df


def shap_stability(bundle) -> pd.DataFrame:
    """Spearman correlation of mean|SHAP| rankings across 5 seeds (stability check)."""
    import shap
    from scipy.stats import spearmanr
    ranks = []
    for seed in SEEDS:
        model = make_model("lightgbm", seed)
        model.fit(bundle.X_both, bundle.y.to_numpy())
        expl = shap.TreeExplainer(model)
        sub = bundle.X_both.sample(n=min(2000, len(bundle.X_both)), random_state=seed)
        sv = expl.shap_values(sub)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        ranks.append(rank)
    taus = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            rho, p = spearmanr(ranks[i], ranks[j])
            taus.append({"seed_i": SEEDS[i], "seed_j": SEEDS[j], "spearman_rho": float(rho),
                         "p": float(p)})
    df = pd.DataFrame(taus)
    df.to_csv(RESULTS_TABLES / "shap_stability.csv", index=False)
    return df


def scenario_probes(bundle) -> pd.DataFrame:
    """Stylized sensitivity: +-10% perturbation of selected descriptor columns,
    bounded within historical range; mean change in predicted Tc (K)."""
    probe_cols = [c for c in ("mean_atomic_mass", "mean_Density", "wtd_range_atomic_mass")
                  if c in bundle.X_both.columns]
    assert probe_cols, "declared probe columns missing from feature set"
    model = make_model("lightgbm", SEED_BASE)
    X = bundle.X_both.copy()
    y = bundle.y.to_numpy()
    model.fit(X, y)
    base_pred = model.predict(X)
    rows = []
    for col in probe_cols:
        lo, hi = X[col].min(), X[col].max()
        for pct in (-10, 10):
            Xp = X.copy()
            Xp[col] = X[col] * (1 + pct / 100)
            Xp[col] = Xp[col].clip(lo, hi)
            dpred = model.predict(Xp) - base_pred
            rows.append({"probe": col, "percent": pct,
                         "mean_delta_pred_Tc_K": float(dpred.mean()),
                         "p05_delta": float(np.quantile(dpred, .05)),
                         "p95_delta": float(np.quantile(dpred, .95)),
                         "clipped_within_historical_range": True})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_TABLES / "scenario_probes.csv", index=False)
    return df


def residual_diagnostics(bundle) -> pd.DataFrame:
    """Residual-as-hypothesis diagnostics on the full-data primary model (seed 0):
    worst over/under-prediction groups by family."""
    model = make_model("lightgbm", SEED_BASE)
    X, y = bundle.X_both, bundle.y.to_numpy()
    model.fit(X, y)
    pred = model.predict(X)
    df = pd.DataFrame({"formula": bundle.formulas, "family": bundle.families,
                       "Tc_K": y, "pred_K": pred, "residual_K": y - pred})
    df.to_csv(RESULTS_TABLES / "full_fit_residuals.csv", index=False)
    fam_res = df.groupby("family")["residual_K"].agg(["mean", "std", "count"]).round(3)
    fam_res.to_csv(RESULTS_TABLES / "residual_by_family.csv")
    worst = df.reindex(df["residual_K"].abs().sort_values(ascending=False).index).head(20)
    worst.to_csv(RESULTS_TABLES / "top20_abs_residuals.csv", index=False)
    return fam_res


def dedup_sensitivity(bundle) -> pd.DataFrame:
    """Supplementary (declared in scope.yaml): collapse multi-record formulas to
    mean Tc; verify feature rows are identical within formula; run iid + family
    regimes with primary + ridge; report as a labelled sensitivity analysis.

    Alignment: all collapsed arrays keep the ORIGINAL row order of first occurrence,
    so features / y / family / formula stay row-aligned (asserted below).
    """
    from train_eval import metrics
    y_all = bundle.y.to_numpy()
    formulas = bundle.formulas.to_numpy()
    fams = bundle.families.to_numpy()
    X = bundle.X_both

    # feature-identity check within formula groups (reported, not altered)
    dup_mask = pd.Series(formulas).duplicated(keep=False).to_numpy()
    g = X[dup_mask].groupby(formulas[dup_mask]).nunique()
    max_variants = int(g.max().max())
    LOG.info("dedup check: max distinct feature-row variants within a formula group = %d "
             "(0 extra variants means all duplicate-feature rows are identical)",
             max_variants - 1)

    # first-occurrence positions per formula, in ORIGINAL row order
    dedup_pos = pd.Series(formulas).drop_duplicates().index.to_numpy()
    # per-row group mean of Tc (aligned to original rows), sampled at dedup positions
    yd = pd.Series(y_all).groupby(formulas).transform("mean").to_numpy()[dedup_pos]
    Xd = X.iloc[dedup_pos].reset_index(drop=True)
    famd = fams[dedup_pos]
    fd = formulas[dedup_pos]

    # hard alignment assertions (features <-> target <-> labels)
    assert len(Xd) == len(yd) == len(famd) == len(fd), "dedup arrays length mismatch"
    assert int(pd.Series(fd).duplicated().sum()) == 0, "duplicate formulas remain after dedup"
    tc_group_check = pd.Series(yd).groupby(fd).transform("mean").to_numpy()
    assert np.allclose(tc_group_check, yd), "dedup target is not constant within formulas"
    LOG.info("dedup collapse: %d rows -> %d unique formulas", len(y_all), len(yd))

    class _DedupBundle:
        pass
    dedup_bundle = _DedupBundle()
    dedup_bundle.X_both, dedup_bundle.X_stat, dedup_bundle.X_elem = Xd, None, None
    dedup_bundle.y = pd.Series(yd)
    dedup_bundle.formulas = pd.Series(fd, name="material")
    dedup_bundle.families = pd.Series(famd, name="family")
    dedup_bundle.elem_psets = [bundle.elem_psets[i] for i in dedup_pos]
    dedup_bundle.elem_cols = bundle.elem_cols

    out_path = RESULTS_TABLES / "dedup_sensitivity.parquet"
    rows, done = _load_done(out_path, lambda r: (r["regime"], r["model"],
                                                 int(r["seed"]), int(r["fold"])))
    for regime in ("iid_random", "family_blocked"):
        for seed, fold, (tr, te) in iter_regime(regime, dedup_bundle, SEEDS, 5):
            for mname in (PRIMARY, BASE_MODEL):
                key = (regime, mname, int(seed), int(fold))
                if key in done:
                    continue
                model = make_model(mname, seed)
                model.fit(Xd.iloc[tr], yd[tr])
                m = metrics(yd[te], model.predict(Xd.iloc[te]))
                rows.append({"regime": regime, "model": mname, "seed": int(seed),
                             "fold": int(fold), "n_train": len(tr), "n_test": len(te),
                             "analysis": "dedup_mean_tc", **m})
                pd.DataFrame(rows).to_parquet(out_path)  # step saving
                done.add(key)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return df
