"""Single entry point (Part 7 standard). Stages:

  gate-b   : preprocessing/split verification report (no training)
  smoke    : tiny 1-seed sanity run of the CV machinery
  main     : full declared CV (4 regimes x 6 models x 5 seeds); supports
             --regimes/--models/--out-name for partitioned (multi-kernel) runs
  summary  : tables + significance tests from saved results
  extras   : ablations, SHAP, scenario probes, residuals, dedup
             (--extras-part 1 = feature-set ablations only;
              --extras-part 2 = top-k removal + SHAP stability + probes
              + residuals + dedup)
  postreview : post-review supplementary analyses
               (family census, weighted means, Mondrian conformal,
               Jaccard distance diagnostic; lightgbm only)
  figures  : journal figures from saved results
  all      : gate-b -> main -> extras -> summary -> figures

Usage: python run_pipeline.py --stage smoke
       python run_pipeline.py --stage main --regimes iid_random,formula_grouped \
           --out-name main_results_a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import SCOPE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gate-b", "smoke", "main", "summary", "extras",
                             "postreview", "figures", "all"])
    ap.add_argument("--regimes", default=None, help="comma list; default = declared scope")
    ap.add_argument("--models", default=None, help="comma list; default = declared scope")
    ap.add_argument("--out-name", default="main_results")
    ap.add_argument("--extras-part", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    if args.stage in ("gate-b", "all"):
        from gate_b_check import main as gate_b
        gate_b()

    if args.stage == "smoke":
        from train_eval import run_main_cv, summarize_save
        df = run_main_cv(regimes=["formula_grouped"],
                         models=["dummy_mean", "ridge"],
                         out_name="smoke_results",
                         seeds_run=[SCOPE["random_seed_base"]],
                         folds_run=3)
        print(summarize_save(df, "smoke_results_summary"))

    if args.stage in ("main", "all"):
        models = args.models.split(",") if args.models else list(SCOPE["models"].keys())
        regimes = (args.regimes.split(",") if args.regimes
                   else list(SCOPE["split_regimes"].keys()))
        from train_eval import run_main_cv
        run_main_cv(regimes=regimes, models=models, out_name=args.out_name)

    if args.stage in ("extras", "all"):
        from features import load_feature_bundle
        from extras import (run_ablations, run_topk_removal, shap_stability,
                            scenario_probes, residual_diagnostics, dedup_sensitivity)
        bundle = load_feature_bundle()
        part = args.extras_part
        if args.stage == "all" or part == 0:
            run_ablations(bundle)
            run_topk_removal(bundle)
            shap_stability(bundle)
            scenario_probes(bundle)
            residual_diagnostics(bundle)
            dedup_sensitivity(bundle)
        elif part == 1:
            run_ablations(bundle)
        elif part == 2:
            run_topk_removal(bundle)
            shap_stability(bundle)
            scenario_probes(bundle)
            residual_diagnostics(bundle)
            dedup_sensitivity(bundle)

    if args.stage in ("summary", "all"):
        import pandas as pd
        from proj_paths import RESULTS_TABLES
        from train_eval import summarize_save, significance_tests
        df = pd.read_parquet(RESULTS_TABLES / "main_results.parquet")
        print(summarize_save(df))
        print(significance_tests(df))

    if args.stage == "postreview":
        from postreview_analysis import main as postreview
        postreview()

    if args.stage == "figures":
        from figures import all_main_figures
        all_main_figures()


if __name__ == "__main__":
    main()
