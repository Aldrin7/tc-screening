"""Feature construction (Gate B). Loads train.csv + unique_m.csv, verifies row alignment,
builds the four declared feature sets, family sieve labels, and element presence sets.

No imputation, no outlier removal, no resampling (none needed; verified at Gate A).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from proj_paths import DATA_KAGGLE

LOG = logging.getLogger("features")

TARGET = "critical_temp"
MATERIAL_COL = "material"


@dataclass
class FeatureBundle:
    X_stat: pd.DataFrame          # 81 statistical descriptors (train.csv features)
    X_elem: pd.DataFrame          # 86 element mass fractions (unique_m.csv)
    X_both: pd.DataFrame          # 167 combined
    y: pd.Series                  # critical_temp (K)
    formulas: pd.Series
    families: pd.Series
    elem_psets: list[set[str]]    # per-row element sets (present elements, frac > 0)
    elem_cols: list[str]
    stat_cols: list[str]


def load_feature_bundle() -> FeatureBundle:
    tr = pd.read_csv(DATA_KAGGLE / "train.csv")
    um = pd.read_csv(DATA_KAGGLE / "unique_m.csv")
    LOG.info("loaded train.csv %s and unique_m.csv %s", tr.shape, um.shape)

    # row alignment: the two files are documented as row-aligned outputs of the same pipeline
    aligned = np.allclose(tr[TARGET].to_numpy(), um[TARGET].to_numpy(), rtol=0, atol=1e-12)
    assert aligned, "train.csv and unique_m.csv are not row-aligned - refusing to join"
    assert len(tr) == 21263 and len(um) == 21263, "unexpected row counts"

    stat_cols = [c for c in tr.columns if c != TARGET]
    elem_cols = [c for c in um.columns if c not in (TARGET, MATERIAL_COL)]
    formulas = um[MATERIAL_COL]

    X_stat = tr[stat_cols].astype(float)
    X_elem = um[elem_cols].astype(float)
    X_both = pd.concat([X_stat.reset_index(drop=True), X_elem.reset_index(drop=True)], axis=1)

    y = tr[TARGET].astype(float)
    fam = _family_labels(X_elem, elem_cols)
    psets = [set(np.array(elem_cols)[np.asarray(fr) > 0]) for fr in X_elem.to_numpy()]

    LOG.info("sample flow: raw %d -> aligned %d -> features stat %d / elem %d / both %d",
             len(tr), len(X_stat), X_stat.shape[1], X_elem.shape[1], X_both.shape[1])
    return FeatureBundle(X_stat, X_elem, X_both, y, formulas, fam, psets, elem_cols, stat_cols)


def _family_labels(X_elem: pd.DataFrame, elem_cols: list[str]) -> pd.Series:
    """Rule-based family sieves (documented heuristics; error-prone by design, flagged).

    Order: elemental -> mgb2 -> cuprate -> iron_pnictide -> heusler_ternary -> other.
    """
    E = X_elem.to_numpy()
    cols = {c: i for i, c in enumerate(elem_cols)}
    n_elements = (E > 0).sum(axis=1)
    fam = np.empty(len(E), dtype=object)
    for i in range(len(E)):
        row = E[i]
        if n_elements[i] == 1:
            fam[i] = "elemental"
        elif n_elements[i] == 2 and row[cols["Mg"]] > 0 and row[cols["B"]] > 0:
            fam[i] = "mgb2_binary"
        elif row[cols["O"]] > 0 and row[cols["Cu"]] > 0:
            fam[i] = "cuprate"
        elif (row[cols["Fe"]] > 0 and
              any(row[cols.get(x, -1)] > 0 for x in ("As", "P", "Se", "S", "Te", "Bi") if x in cols) and
              row[cols["O"]] == 0):
            fam[i] = "iron_pnictide"
        elif n_elements[i] == 3 and row[cols["O"]] == 0:
            fam[i] = "heusler_ternary"
        else:
            fam[i] = "other"
    s = pd.Series(fam, name="family")
    log_counts = s.value_counts()
    LOG.info("family sieve counts: %s", log_counts.to_dict())
    return s
