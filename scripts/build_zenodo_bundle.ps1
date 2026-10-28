# Builds results/zenodo_bundle/ : curated, upload-ready deposit for Zenodo.
# v2 layout: the deposit mirrors the tracked repository tree (code, configs,
# data manifest, results) exactly as published on GitHub, plus the out-of-fold
# prediction table shipped as 2 MiB raw byte parts (single large PUTs do not
# survive the congested route to zenodo.org; Zenodo's multipart API is broken
# upstream, zenodo-rdm#1443). The manuscript is not part of the deposit: it is
# submitted to the journal and copyright passes to the publisher on acceptance.
# Usage:  powershell -File scripts\build_zenodo_bundle.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$Commit = (git rev-parse HEAD).Trim()

$out = Join-Path $root 'results\zenodo_bundle'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path $out -Force | Out-Null

# 1) every tracked file = the GitHub tree (secrets and large artifacts are
#    gitignored and never tracked)
git ls-files | ForEach-Object {
  $dst = Join-Path $out $_
  $dir = Split-Path -Parent $dst
  if ($dir) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Copy-Item $_ $dst
}

# 2) out-of-fold predictions: split into raw byte parts (reassembly = concatenation)
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

# 3) MANIFEST.txt with SHA-256 + sizes (integrity layer)
$mf = Join-Path $out 'MANIFEST.txt'
"Zenodo bundle MANIFEST - TcScreening" | Out-File $mf -Encoding utf8
"Generated: $stamp   Source commit: $Commit   Repository: https://github.com/Aldrin7/tc-screening" | Out-File $mf -Append -Encoding utf8
"" | Out-File $mf -Append -Encoding utf8
"Contents: the full tracked repository tree (code, configs, data provenance manifest," | Out-File $mf -Append -Encoding utf8
"result tables, figures, run logs) plus the out-of-fold prediction parts below." | Out-File $mf -Append -Encoding utf8
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
