# Builds results/zenodo_bundle/ : curated, upload-ready deposit for Zenodo.
# Usage:  powershell -File scripts\build_zenodo_bundle.ps1 [-Commit <sha>]
param(
  [string]$Commit = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Commit) { $Commit = (git rev-parse HEAD).Trim() }
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$out = Join-Path $root 'results\zenodo_bundle'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path "$out\results\tables" -Force | Out-Null
New-Item -ItemType Directory -Path "$out\results\figures" -Force | Out-Null
New-Item -ItemType Directory -Path "$out\results\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "$out\configs" -Force | Out-Null

# 1) code snapshot (tracked files only; secrets/data/large artifacts never tracked)
$zip = Join-Path $out "tc-screening-repo-$($Commit.Substring(0,8)).zip"
git archive --format=zip --prefix "tc-screening/" -o $zip $Commit
Write-Host "code snapshot: $zip"

# 2) canonical result tables (paper data; smoke_* are pipeline evidence only -> excluded)
#    main_predictions.parquet is NOT copied directly: a single 34.5 MB PUT does not
#    survive the congested route to zenodo.org (mid-body teardowns observed 2026-10-27;
#    Zenodo's multipart API is broken upstream, zenodo-rdm#1443). It is shipped as
#    2 MiB raw byte parts that reassemble byte-identically (see MANIFEST note).
$tables = @(
  'main_results.parquet','main_results.csv','main_results_summary.csv','main_results_summary.md',
  'significance_tests.csv',
  'ablation_feature_sets.parquet','ablation_topk_removal.parquet',
  'dedup_sensitivity.parquet','scenario_probes.csv',
  'shap_stability.csv','residual_by_family.csv','top20_abs_residuals.csv',
  'full_fit_residuals.csv',
  'postreview_family_census.csv','postreview_weighted_means.csv',
  'postreview_mondrian.csv','postreview_jaccard.csv','postreview_norm_ablation.csv'
)
foreach ($f in $tables) {
  $src = Join-Path $root "results\tables\$f"
  if (Test-Path $src) { Copy-Item $src "$out\results\tables\" } else { Write-Warning "missing table: $f" }
}

# 2b) split main_predictions.parquet into raw byte parts (reassembly = concatenation)
$predSrc = Join-Path $root 'results\tables\main_predictions.parquet'
if (-not (Test-Path $predSrc)) { throw "missing: $predSrc" }
$predSize = (Get-Item $predSrc).Length
$predHash = (Get-FileHash $predSrc -Algorithm SHA256).Hash.ToLower()
$predBytes = [IO.File]::ReadAllBytes($predSrc)
$chunk = 2MB
$nParts = [int][math]::Ceiling($predBytes.Length / $chunk)
$partsDir = Join-Path $out 'results\tables\main_predictions_parts'
New-Item -ItemType Directory -Path $partsDir -Force | Out-Null
for ($i = 0; $i -lt $nParts; $i++) {
  $lo = $i * $chunk
  $len = [math]::Min($chunk, $predBytes.Length - $lo)
  $part = New-Object byte[] $len
  [Array]::Copy($predBytes, $lo, $part, 0, $len)
  [IO.File]::WriteAllBytes((Join-Path $partsDir ("part_{0:d2}of{1}.bin" -f ($i + 1), $nParts)), $part)
}
Remove-Variable predBytes
Write-Host "split main_predictions.parquet ($predSize B) -> $nParts parts of <= 2 MiB"
# SHAP full ranking lives in future_scope/
$shap = Join-Path $root 'future_scope\shap_mean_abs_full_order.csv'
if (Test-Path $shap) { Copy-Item $shap "$out\results\tables\shap_mean_abs_full_order.csv" }

# 3) figures (PDF vector + PNG)
Copy-Item "$root\results\figures\*" "$out\results\figures\"

# 4) provenance logs
foreach ($f in @('kaggle_run_manifest_a.json','kaggle_run_manifest_b.json','run_manifest.json',
                 'main_run.txt','summary_run.txt','figures_run3.txt','postreview_manifest.json')) {
  $src = Join-Path $root "results\logs\$f"
  if (Test-Path $src) { Copy-Item $src "$out\results\logs\" }
}

# 5) scope + requirements
Copy-Item "$root\configs\scope.yaml" "$out\configs\scope.yaml"
Copy-Item "$root\requirements.txt" "$out\requirements.txt"

# 6) MANIFEST.txt with SHA-256 + sizes (integrity layer)
$mf = Join-Path $out 'MANIFEST.txt'
"Zenodo bundle MANIFEST - TcScreening" | Out-File $mf -Encoding utf8
"Generated: $stamp   Source commit: $Commit   Repository: https://github.com/Aldrin7/tc-screening" | Out-File $mf -Append -Encoding utf8
"" | Out-File $mf -Append -Encoding utf8
"Reassembly note: results/tables/main_predictions.parquet (the 3,786,600-row out-of-fold" | Out-File $mf -Append -Encoding utf8
"prediction table, $predSize bytes) is distributed as $nParts raw byte parts under" | Out-File $mf -Append -Encoding utf8
"results/tables/main_predictions_parts/. Reassemble by concatenating the parts in filename" | Out-File $mf -Append -Encoding utf8
"order, e.g.:  cat main_predictions_parts/part_*.bin > main_predictions.parquet   (bash)" | Out-File $mf -Append -Encoding utf8
"or:           copy /b main_predictions_parts\part_00of$nParts.bin+...+main_predictions_parts\part_XXof$nParts.bin main_predictions.parquet   (cmd)" | Out-File $mf -Append -Encoding utf8
"Integrity check: the concatenated file must be exactly $predSize bytes with" | Out-File $mf -Append -Encoding utf8
"SHA-256 = $predHash" | Out-File $mf -Append -Encoding utf8
"" | Out-File $mf -Append -Encoding utf8
Get-ChildItem $out -Recurse -File | Where-Object { $_.Name -ne 'MANIFEST.txt' } | ForEach-Object {
  $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
  "{0}  {1,12:N0} B  {2}" -f $h, $_.Length, ($_.FullName.Substring($out.Length+1)) | Out-File $mf -Append -Encoding utf8
}
Write-Host "bundle ready: $out"
Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum | ForEach-Object { "total {0:N1} MB, {1} files" -f ($_.Sum/1MB), $_.Count }
