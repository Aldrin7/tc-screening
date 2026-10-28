# One-command Zenodo deposit for the TcScreening bundle (Zenodo RDM / InvenioRDM API).
# This is a thin wrapper around scripts\upload_to_zenodo.py, which implements the
# actual RDM flow (init [{"key":...}] -> PUT content -> commit; PS 5.1 HTTP stacks
# are rejected by Zenodo's fronting proxy, so the real work runs on Python urllib).
# Prereq: a Zenodo PERSONAL ACCESS token with write scopes
#         (https://zenodo.org/account/settings/applications/tokens/ - New token:
#          tick "deposit:write" and "deposit:actions"; NOT the OAuth applications
#          section and NOT sandbox.zenodo.org - neither works against production).
# Usage:
#   powershell -File scripts\upload_to_zenodo.ps1 -Token "YOUR_TOKEN"             # draft+upload+publish
#   powershell -File scripts\upload_to_zenodo.ps1 -Token "YOUR_TOKEN" -NoPublish  # keep as private draft
#   powershell -File scripts\upload_to_zenodo.ps1 -Token "YOUR_TOKEN" -DraftId 23686822 -NoPublish  # resume
param(
  [Parameter(Mandatory = $true)][string]$Token,
  [string]$DraftId,
  [switch]$NoPublish
)
$ErrorActionPreference = 'Stop'
$py = Join-Path $PSScriptRoot 'upload_to_zenodo.py'
$argList = @($py, '--token', $Token)
if ($DraftId) { $argList += @('--draft-id', $DraftId) }
if (-not $NoPublish) { $argList += '--publish' }
& python @argList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
