#!/usr/bin/env python
"""Zenodo RDM deposit uploader for the TcScreening bundle.

Implements the current Zenodo (InvenioRDM) REST API flow, verified against the
production API in Oct 2026:
  - draft create/resume:   POST /api/records | GET /api/records/<id>/draft
  - files init:            POST <draft>/files            body: [{"key": "<name>"}]
  - bytes:                 PUT  entry.links.content      (octet-stream)
  - commit:                POST <draft>/files/<urlenc key>/commit
  - publish:               POST draft.links.publish      (irreversible; mints the DOI)

The token is read from a file (never echoed, never logged). Transient 5xx
responses are retried with backoff - Zenodo's API is occasionally flaky.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://zenodo.org/api/records'
AFFIL = 'ICFAI Tech School, The ICFAI University, Dehradun, Uttarakhand, India'

DESC = (
    '<p>Complete result set and code for the study "Composition-based screening of superconducting '
    'critical temperature: an interpretable, uncertainty-calibrated machine-learning framework '
    'evaluated under leakage-controlled validation" (Materials Futures, Focus Issue: '
    'Artificial Intelligence for Materials Science).</p>'
    '<p>Contents: 630 controlled fits (4 split regimes &times; 6 models &times; 5 seeds &times; 5 folds; '
    'RMSE/MAE/R<sup>2</sup>, split-conformal 90% coverage and widths), Holm-corrected paired '
    'significance tests, feature-set and top-k ablations, deduplication sensitivity, SHAP '
    'attributions and stability, stylized perturbation probes, residuals, 3,786,600 out-of-fold '
    'prediction rows, post-review analyses (family census, test-size-weighted means, Mondrian '
    'conformal, instance-distance diagnostic, normalization ablation), run manifests with package '
    'versions and scope SHA-256, paper figures (vector PDF + PNG), and a code snapshot (MIT licence).</p>'
    '<p>Source data: SuperCon database (NIMS) as distributed by Hamidieh (2018) via the UCI '
    'Machine Learning Repository, DOI 10.24432/C53P47 (CC BY 4.0) - not redistributed here.</p>'
)

METADATA = {
    'metadata': {
        'title': 'TcScreening: leakage-aware, uncertainty-calibrated machine-learning screening of '
                 'superconducting critical temperature - full results and code',
        'resource_type': {'id': 'dataset'},
        'publication_date': time.strftime('%Y-%m-%d'),
        # NOTE (verified 2026-10-27 against production): this Zenodo writes creators
        # only with singular family_name/given_name AND the computed 'name' inside
        # person_or_org; type must be 'personal' (not 'person'). relation_type ids are
        # lowercase. Writes need 'Accept: application/vnd.inveniordm.v1+json' or the
        # service legacy-normalizes the payload (strips person_or_org/relation_type).
        'creators': [{'person_or_org': {'type': 'personal', 'family_name': 'Manon',
                                        'given_name': 'Aldrin', 'name': 'Manon, Aldrin'},
                      'affiliations': [{'name': AFFIL}]}],
        'publisher': 'Aldrin Manon',
        'description': DESC,
        'rights': [{'id': 'cc-by-4.0',
                    'title': {'en': 'Creative Commons Attribution 4.0 International'}}],
        'subjects': [{'subject': s} for s in
                     ['superconductors', 'critical temperature', 'machine learning',
                      'data leakage', 'conformal prediction', 'uncertainty quantification']],
        'related_identifiers': [
            {'identifier': '10.24432/C53P47', 'scheme': 'doi',
             'relation_type': {'id': 'isderivedfrom'}, 'resource_type': {'id': 'dataset'}},
            {'identifier': 'https://github.com/Aldrin7/tc-screening', 'scheme': 'url',
             'relation_type': {'id': 'issupplementto'}, 'resource_type': {'id': 'software'}},
        ],
    }
}


def call(method: str, url: str, token: str, *, body: dict | list | None = None,
         raw: bytes | None = None, content_type: str = 'application/json',
         retries: int = 4, timeout: int = 300):
    """One HTTP call with retry-on-5xx. Returns (status, parsed_json_or_None)."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Authorization', 'Bearer ' + token)
        req.add_header('User-Agent', 'tc-screening-uploader/1.0')
        if data is not None:
            req.add_header('Content-Type', content_type)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return r.status, (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as e:
            payload = e.read()
            if 500 <= e.code < 600 and attempt < retries:      # server-side flake: back off, retry
                wait = 5 * (attempt + 1)
                print(f'  ! {e.code} on {method} {url} - retrying in {wait}s', flush=True)
                time.sleep(wait)
                last = e
                continue
            print(f'  ! {e.code} on {method} {url}: {payload[:300]!r}', flush=True)
            return e.code, (json.loads(payload) if payload else None)
        except Exception as e:                                  # network error: retry
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f'  ! {e!r} - retrying in {wait}s', flush=True)
                time.sleep(wait)
                last = e
                continue
            print(f'  ! {e!r} on {method} {url}', flush=True)
            return None, None
    raise RuntimeError(f'unreachable; last error: {last}')


def bundle_files(bundle: Path) -> list[tuple[str, Path]]:
    files = sorted(p for p in bundle.rglob('*') if p.is_file())
    return [(p.relative_to(bundle).as_posix(), p) for p in files]


def put_file(url: str, path: Path, token: str, attempts: int = 4) -> int:
    """PUT a file via curl -T (streams from disk, sends Expect: 100-continue).

    Python's urllib died with SSLEOFError on the ~40 MB parquet (Zenodo's fronting
    proxy tears down long non-100-continue uploads); curl handles this correctly.
    The token goes through a temp curl config file, never through ARGV."""
    code = None
    for attempt in range(attempts):
        cfg = tempfile.NamedTemporaryFile('w', suffix='.curlcfg', delete=False)
        cfg.write(f'header = "Authorization: Bearer {token}"\n')
        cfg.close()
        try:
            p = subprocess.run(
                ['curl.exe', '-sS', '-X', 'PUT', '-H', 'Content-Type: application/octet-stream',
                 '-K', cfg.name, '-T', str(path), '-o', os.devnull,
                 '-w', '%{http_code} %{speed_upload}B/s %{time_total}s',
                 '--connect-timeout', '30', '--max-time', '3600',
                 '--speed-limit', '2048', '--speed-time', '45', url],
                capture_output=True, text=True, timeout=3700)
            out = p.stdout.strip()
            parts = out.split()
            code = int(parts[0]) if parts and parts[0].isdigit() else None
            info = ' '.join(parts[1:]) if len(parts) > 1 else ''
        except Exception as e:
            print(f'  ! curl PUT error: {e!r}', flush=True)
            code, info = None, ''
        finally:
            try:
                os.unlink(cfg.name)
            except OSError:
                pass
        if code in (200, 201):
            if info:
                print(f'  PUT ok: {info}', flush=True)
            return code
        wait = 10 * (attempt + 1)
        print(f'  ! PUT HTTP {code} {info} - retrying in {wait}s', flush=True)
        time.sleep(wait)
    return code or 0


def upload_one(rel: str, path: Path, files_url: str, token: str, attempts: int) -> bool:
    """Upload one file with patient retry: on any failure, drop the (possibly
    half-written) remote entry and start over with a fresh init/content URL.

    The route to zenodo.org intermittently presents wrong TLS certificates and
    tears down mid-body uploads (observed 2026-10-27, both IPv4 and IPv6);
    successful windows do occur, so the loop just keeps trying."""
    quote = lambda k: urllib.parse.quote(k, safe='')
    for attempt in range(1, attempts + 1):
        code, listing = call('GET', files_url, token)
        if code == 200 and listing:
            for e in (listing.get('entries') or []):
                if e.get('key') == rel:                       # stale/pending entry from a failed try
                    call('DELETE', f'{files_url}/{quote(rel)}', token)
                    break
        code, init = call('POST', files_url, token, body=[{'key': rel}])
        if code not in (200, 201):
            print(f'  init failed for {rel} (HTTP {code})', flush=True)
            time.sleep(15)
            continue
        entries = init.get('entries') or []
        content_url = next((e['links']['content'] for e in entries if e.get('key') == rel), None)
        if not content_url:
            print(f'  no content link for {rel}: {json.dumps(init)[:200]}', flush=True)
            time.sleep(15)
            continue
        code = put_file(content_url, path, token, attempts=1)
        if code not in (200, 201):
            print(f'  ! PUT attempt {attempt}/{attempts} for {rel} failed (HTTP {code})', flush=True)
            time.sleep(30 if path.stat().st_size > 10_000_000 else 10)
            continue
        code, _ = call('POST', f'{files_url}/{quote(rel)}/commit', token)
        if code not in (200, 201):
            print(f'  commit failed for {rel} (HTTP {code})', flush=True)
            time.sleep(15)
            continue
        if attempt > 1:
            print(f'  (succeeded on attempt {attempt})', flush=True)
        return True
    return False


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--token-file', default=str(repo.parent / 'Zenodo token.txt'),
                    help='file containing the Zenodo personal access token (default: repo-adjacent)')
    ap.add_argument('--token', default=None,
                    help='token string (alternative to --token-file)')
    ap.add_argument('--bundle', default=str(repo / 'results' / 'zenodo_bundle'))
    ap.add_argument('--draft-id', type=int, default=None, help='resume an existing draft')
    ap.add_argument('--publish', action='store_true',
                    help='publish the deposit after successful upload (mints the DOI, irreversible)')
    args = ap.parse_args()

    token = (args.token or Path(args.token_file).read_text(encoding='utf-8-sig')).strip()
    if not token:
        print('token file empty', file=sys.stderr)
        return 2
    bundle = Path(args.bundle)
    files = bundle_files(bundle)
    if not files:
        print(f'no files in {bundle} - run scripts/build_zenodo_bundle.ps1 first', file=sys.stderr)
        return 2
    total_mb = sum(p.stat().st_size for _, p in files) / 1e6
    print(f'bundle: {len(files)} files, {total_mb:.1f} MB')

    # 1) draft: resume or create
    if args.draft_id:
        code, draft = call('GET', f'{API}/{args.draft_id}/draft', token, retries=1)
        if code != 200:
            # Zenodo's draft-read endpoint can 500 while the files endpoints stay
            # healthy (observed 2026-10-27). All URLs we need are constructible:
            print(f'draft read HTTP {code} - proceeding with constructed URLs')
            draft = {'id': args.draft_id,
                     'links': {'self': f'{API}/{args.draft_id}/draft',
                               'files': f'{API}/{args.draft_id}/draft/files',
                               'publish': f'{API}/{args.draft_id}/draft/actions/publish',
                               'self_html': f'https://zenodo.org/uploads/{args.draft_id}'}}
        else:
            print(f'resumed draft {draft["id"]}')
    else:
        code, draft = call('POST', API, token, body=METADATA)
        if code not in (200, 201):
            print(f'draft creation failed (HTTP {code})', file=sys.stderr)
            return 1
        print(f'draft created: id={draft["id"]}  status={draft.get("status")}')

    self_url = draft['links']['self']
    files_url = draft['links']['files']

    # 2) inventory + stray cleanup
    code, listing = call('GET', files_url, token)
    remote: dict[str, int] = {}
    if code == 200 and listing:
        for e in listing.get('entries') or []:
            remote[e['key']] = int(e.get('size') or 0)
    local = dict(files)
    for key in list(remote):
        if key not in local:
            code, _ = call('DELETE', f'{files_url}/{urllib.parse.quote(key, safe="")}', token)
            print(f'  removed stray remote entry: {key} (HTTP {code})')
            remote.pop(key)

    # 3) upload (init -> PUT content -> commit), skipping files already present
    #    with matching size; big files get many patient attempts because the
    #    route to zenodo.org intermittently breaks mid-upload (see upload_one)
    for rel, path in files:
        size = path.stat().st_size
        if remote.get(rel) == size:
            print(f'  already uploaded: {rel}')
            continue
        attempts = 60 if size > 10_000_000 else 8
        if not upload_one(rel, path, files_url, token, attempts):
            print(f'gave up on {rel} after {attempts} attempts', file=sys.stderr)
            return 1
        print(f'  uploaded: {rel} ({size / 1e6:.2f} MB)', flush=True)

    # 4) verify
    code, listing = call('GET', files_url, token)
    remote = {e['key']: int(e.get('size') or 0) for e in (listing.get('entries') or [])} \
        if code == 200 and listing else {}
    missing = [rel for rel, p in files if remote.get(rel) != p.stat().st_size]
    if missing:
        print(f'verification failed - missing/mismatched: {missing}', file=sys.stderr)
        return 1
    print(f'verified: all {len(files)} files present with matching sizes')

    # 5) publish (irreversible)
    if not args.publish:
        print(f'draft kept unpublished: {draft["links"].get("self_html", "(no html link)")}')
        print(f'publish later with: --draft-id {draft["id"]} --publish')
        return 0

    publish_url = draft['links'].get('publish') or f'{self_url}/actions/publish'
    code, pub = call('POST', publish_url, token)
    if code not in (200, 201):
        print(f'publish failed (HTTP {code})', file=sys.stderr)
        return 1
    print(f'PUBLISHED. DOI: {pub.get("doi")}')
    print(f'Record: {pub.get("links", {}).get("self_html")}')
    print('Fill this DOI into docs/manuscript.md (Data availability) and docs/cover_letter.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
