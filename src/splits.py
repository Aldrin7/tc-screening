"""Split-regime builders (Gate B). Four declared regimes in configs/scope.yaml.

Every regime yields, per (seed, fold), integer index arrays (train_idx, test_idx).
Leakage assertions are performed here so that a wrong split fails loudly, never silently.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold, KFold

from features import FeatureBundle, TARGET

LOG = logging.getLogger("splits")

Idx = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]


def _all_row_indices(n: int) -> npt.NDArray[np.int64]:
    return np.arange(n, dtype=np.int64)


def iter_regime(regime: str, bundle: FeatureBundle, seeds: list[int],
                folds: int) -> Iterator[tuple[int, int, Idx]]:
    """Yield (seed, fold_index, (train_idx, test_idx)) for the named regime."""
    n = len(bundle.y)
    if regime == "iid_random":
        for seed in seeds:
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(kf.split(_all_row_indices(n))):
                yield seed, f, (np.asarray(tr), np.asarray(te))
    elif regime == "formula_grouped":
        groups = bundle.formulas.to_numpy()
        for seed in seeds:
            gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(gkf.split(_all_row_indices(n), groups=groups)):
                leak = set(groups[tr]) & set(groups[te])
                assert not leak, f"formula leakage in fold ({seed},{f}): {len(leak)} shared formulas"
                yield seed, f, (np.asarray(tr), np.asarray(te))
    elif regime == "family_blocked":
        fams = bundle.families.to_numpy()
        fam_names = sorted(set(fams))
        assert len(fam_names) >= 5, f"expected >=5 families, got {fam_names}"
        for fold, held_out in enumerate(fam_names):
            te = np.where(fams == held_out)[0]
            tr = np.where(fams != held_out)[0]
            # seed does not change fold composition here; it changes model randomness
            for seed in seeds:
                yield seed, fold, (np.asarray(tr), np.asarray(te))
    elif regime == "element_leaveout":
        rows = np.arange(n)
        counts = {c: int((np.asarray([c in s for s in bundle.elem_psets])).sum())
                  for c in bundle.elem_cols}
        eligible = [c for c, cnt in counts.items() if cnt >= 100]
        assert len(eligible) >= 25, f"too few eligible elements for blocking: {len(eligible)}"
        for seed in seeds:
            rng = np.random.default_rng(seed)
            blocks = np.array_split(rng.permutation(sorted(eligible)), folds)
            for f, block in enumerate(blocks):
                block_set = set(block)
                in_test = np.array([bool(pset & block_set) for pset in bundle.elem_psets])
                te = rows[in_test]
                tr = rows[~in_test]
                leak = (np.array([c in pset for pset in
                                  [bundle.elem_psets[i] for i in tr]
                                  for c in block])).any() if len(tr) else False
                assert not leak, f"element leakage in block ({seed},{f})"
                yield seed, f, (np.asarray(tr), np.asarray(te))
    else:
        raise ValueError(f"unknown regime {regime}")


def fold_counts(regime: str, bundle: FeatureBundle, seeds: list[int]) -> dict:
    """Report fold sizes without materializing training (Gate B logging)."""
    info = {}
    n = len(bundle.y)
    for seed, fold, (tr, te) in iter_regime(regime, bundle, seeds, 5):
        key = f"s{seed}f{fold}"
        info[key] = (int(len(tr)), int(len(te)))
    lens_tr = [v[0] for v in info.values()]
    lens_te = [v[1] for v in info.values()]
    return {"n": n, "n_folds_seeds": len(info),
            "train_min": min(lens_tr), "train_max": max(lens_tr),
            "test_min": min(lens_te), "test_max": max(lens_te)}
