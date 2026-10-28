"""Gate D figures. Reads results tables, produces journal-grade figures (Arial, >=300 DPI,
explicit number formatting). Only plots what exists in results/tables."""
from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proj_paths import RESULTS_TABLES, RESULTS_FIGURES

LOG = logging.getLogger("figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.formatter.useoffset": False,
    "savefig.dpi": 600, "figure.dpi": 150,})


def _save(fig: plt.Figure, fname: str) -> None:
    """Save every figure as PNG (preview) and PDF (vector, venue-submittable)."""
    stem = fname.rsplit(".", 1)[0]
    fig.savefig(RESULTS_FIGURES / fname, bbox_inches="tight")
    fig.savefig(RESULTS_FIGURES / f"{stem}.pdf", bbox_inches="tight")

MODEL_ORDER = ["dummy_mean", "ridge", "random_forest", "xgboost", "lightgbm", "catboost"]
MODEL_LABEL = {"dummy_mean": "Mean", "ridge": "Ridge", "random_forest": "RF",
               "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}
REGIME_LABEL = {"iid_random": "Random (iid)", "formula_grouped": "Formula-grouped",
                "family_blocked": "Family-blocked", "element_leaveout": "Element leave-out"}
REGIME_ORDER = ["iid_random", "formula_grouped", "family_blocked", "element_leaveout"]


def _grouped_bars(ax, values, errs, n_groups, n_bars, palette=None):
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    for b in range(n_bars):
        ax.bar(x + b * width, values[b], width, yerr=errs[b], capsize=2,
               color=palette[b] if palette else None, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (n_bars - 1) / 2)


def figure_rmse_by_regime(df: pd.DataFrame, fname: str = "fig_rmse_by_regime.png") -> None:
    """RMSE mean +- std across (seed, fold), models x regimes, grouped bars."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    regimes = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    means, errs = [], []
    for m in models:
        sel = df[(df.model == m)]
        vals = [sel[sel.regime == r]["rmse_K"].mean() for r in regimes]
        stds = [sel[sel.regime == r]["rmse_K"].std() for r in regimes]
        means.append(vals)
        errs.append(stds)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _grouped_bars(ax, np.array(means), np.array(errs), len(regimes), len(models))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes])
    ax.set_ylabel("RMSE (K)")
    ax.legend([MODEL_LABEL[m] for m in models], ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def figure_coverage(df: pd.DataFrame, fname: str = "fig_conformal_coverage.png") -> None:
    """Measured 90%-nominal conformal coverage per regime (primary model),
    nominal line at 0.90."""
    primary = [m for m in MODEL_ORDER if m in set(df["model"])][-1]
    sel = df[df.model == primary].groupby("regime")["coverage_90"].agg(["mean", "std"])
    regimes = [r for r in REGIME_ORDER if r in sel.index]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    vals = [sel.loc[r, "mean"] for r in regimes]
    stds = [sel.loc[r, "std"] for r in regimes]
    ax.bar(np.arange(len(regimes)), vals, yerr=stds, capsize=3,
           color="#4878CF", edgecolor="black", linewidth=0.5)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
    ax.text(len(regimes) - 0.4, 0.906, "nominal 90%", ha="right", fontsize=8)
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes], rotation=20, ha="right")
    ax.set_ylabel(f"measured coverage ({MODEL_LABEL[primary]}, 90% conformal)")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def figure_regression_scatter(bundle_preds: pd.DataFrame, fname: str = "fig_pred_vs_actual.png"):
    """pred vs actual scatter (primary model, iid + grouped), line y=x."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for ax, regime, label in zip(axes, ("iid_random", "formula_grouped"),
                                 ("Random (iid)", "Formula-grouped")):
        sub = bundle_preds[bundle_preds.regime == regime]
        ax.scatter(sub["Tc_K"], sub["pred_K"], s=3, alpha=0.25, linewidths=0)
        lim = max(sub["Tc_K"].max(), sub["pred_K"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("actual Tc (K)")
    axes[0].set_ylabel("predicted Tc (K)")
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def figure_tc_distribution(df: pd.DataFrame, fname: str = "fig_tc_distribution.png"):
    tr = pd.read_csv(RESULTS_TABLES.parent / ".." / "data" / "raw" / "supercon_kaggle" / "train.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.hist(tr["critical_temp"], bins=60, color="#4878CF", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("critical temperature (K)")
    ax.set_ylabel("number of records")
    ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def figure_scatter_from_predictions(fname: str = "fig_pred_vs_actual.png") -> None:
    """Out-of-fold pred-vs-actual scatter (lightgbm, seed 20260101), iid vs grouped,
    from the merged per-row predictions parquet."""
    preds = pd.read_parquet(RESULTS_TABLES / "main_predictions.parquet")
    sel = preds[(preds.model == "lightgbm") & (preds.seed == preds.seed.min())
                & (preds.regime.isin(["iid_random", "formula_grouped"]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for ax, regime, label in zip(axes, ("iid_random", "formula_grouped"),
                                 ("Random (iid)", "Formula-grouped")):
        sub = sel[sel.regime == regime]
        ax.scatter(sub["y_true"], sub["y_pred"], s=3, alpha=0.2, linewidths=0)
        lim = max(sub["y_true"].max(), sub["y_pred"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("actual Tc (K)")
    axes[0].set_ylabel("predicted Tc (K)")
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def figure_coverage_by_family(df: pd.DataFrame,
                              fname: str = "fig_coverage_by_regime_model.png") -> None:
    """Measured 90%-nominal conformal coverage for all models x regimes."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique() and m != "dummy_mean"]
    regimes = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    means, errs = [], []
    for m in models:
        sel = df[df.model == m]
        means.append([sel[sel.regime == r]["coverage_90"].mean() for r in regimes])
        errs.append([sel[sel.regime == r]["coverage_90"].std() for r in regimes])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _grouped_bars(ax, np.array(means), np.array(errs), len(regimes), len(models))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes])
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
    ax.text(len(regimes) - 0.45, 0.912, "nominal 90%", ha="right", fontsize=8)
    ax.set_ylabel("measured coverage (90% conformal)")
    ax.set_ylim(0.0, 1.0)
    ax.legend([MODEL_LABEL[m] for m in models], ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    LOG.info("wrote %s", fname)


def all_main_figures() -> None:
    df = pd.read_parquet(RESULTS_TABLES / "main_results.parquet")
    figure_tc_distribution(df)
    figure_rmse_by_regime(df)
    figure_coverage(df)
    figure_coverage_by_family(df)
    try:
        figure_scatter_from_predictions()
    except Exception as exc:
        LOG.warning("scatter figure skipped: %s", exc)
