# TcScreening

Leakage-aware screening of superconducting critical temperature (T<sub>c</sub>) from
chemical composition: an interpretable, uncertainty-calibrated machine-learning
framework evaluated under leakage-controlled validation.

**Author:** Aldrin Manon — ICFAI Tech School, The ICFAI University, Dehradun,
Uttarakhand 248011, India — aldrinmanon7@gmail.com

## What this is

An end-to-end pipeline over the standard 21,263-record SuperCon-derived composition
dataset: four validation regimes (random, formula-grouped, family-blocked,
element-extrapolated) crossed with six models (mean baseline, ridge, random forest,
XGBoost, LightGBM, CatBoost) over five seeds and five folds, with a split-conformal
90 % prediction interval attached to every fit — 630 controlled fits in total.
SHAP attributions with stability analysis, ±10 % scenario probes, duplicate
sensitivity analyses and a distance-to-training diagnostic are included.

Headline result: the split protocol moves RMSE by 3.4× (9.4 K under random splits →
31.5 K under element extrapolation) while model choice moves it by at most 1.6 K;
conformal coverage collapses from 0.90 to 0.73 (family-blocked) and 0.49
(element-extrapolated) while the intervals narrow. The pre-registered grid, seeds
(20260101–20260105) and fold structure are fixed in `configs/scope.yaml` (SHA-256
recorded in the run manifests); results are checkpointed per fit, so interrupted
runs resume.

## Repository layout

```
configs/        scope.yaml (normative scope declaration)
src/            data checks, splits, models, training/eval, conformal, SHAP, figures
scripts/        Zenodo bundle builder and uploader
data/curation/  dataset provenance manifest (raw data fetched from UCI, CC BY 4.0)
results/
  tables/       main results, significance tests, ablations, dedup, probes, SHAP,
                post-review analyses (census, weighted means, Mondrian, Jaccard)
  figures/      5 paper figures as PNG (600 dpi) + vector PDF
  logs/         run manifests and stage logs
future_scope/   full SHAP mean-|contribution| ranking
```

## Reproduction

```bash
pip install -r requirements.txt
python run_pipeline.py --stage gate-b      # dataset verification (dual mirror)
python run_pipeline.py --stage smoke       # 6-fit sanity check
python run_pipeline.py --stage main        # full cross-validation (long)
python run_pipeline.py --stage summary     # merge + Holm-corrected tests
python run_pipeline.py --stage extras      # ablations, top-k removal, SHAP, probes, dedup
python run_pipeline.py --stage postreview  # supplementary analyses
python run_pipeline.py --stage figures     # PNG + PDF figures
```

Local Python 3.11; the main metrics were computed on Kaggle CPU (environment and
package versions recorded in `results/logs/kaggle_run_manifest_{a,b}.json`).

## Data

SuperCon database (NIMS) as distributed by Hamidieh (2018) via the UCI Machine
Learning Repository, DOI 10.24432/C53P47, CC BY 4.0. The raw table is not
redistributed here; `data/curation/data_manifest.json` records the provenance and
verification of the two ingested mirrors.

All result tables, the 3.79 M-row out-of-fold prediction table and the run
manifests are archived on Zenodo: https://doi.org/10.5281/zenodo.23686973

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 (UCI).
