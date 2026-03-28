# Security & Deployment Audit — notes_web_builder

**Module version:** post-audit  
**Audited against:** original `notes_web_builder.py` (v1, pre-fix)

---

## Findings & Fixes

### P0 — Security (exploitable)

| # | Finding | Location | Fix applied |
|---|---------|----------|-------------|
| S1 | **javascript: URL injection** — `_attr()` HTML-escapes quote characters but does NOT validate the URL scheme. An article with `"url": "javascript:alert(1)"` would render as a live executable link in any browser. | `_article_card()` L~203 | Added `_safe_url()` — whitelists `http`, `https`, and relative URLs only. Falls back to `#` silently. Also added `referrerpolicy="no-referrer"` to all external links. |
| S2 | **Path traversal via `--date`** — `date_str` was used directly in `Path / "notes" / date_str` with no format check. `--date ../../etc/passwd` would traverse the filesystem on any runner or server. | `load_notes()`, `main()` | Added `_validate_date()` using a strict `^\d{4}-(?:0[1-9]\|1[0-2])-(?:0[1-9]\|[12]\d\|3[01])$` regex. Called at load time AND in the shell step of the workflow for defence-in-depth. |

---

### P1 — Logic bugs (silent bad output)

| # | Finding | Location | Fix applied |
|---|---------|----------|-------------|
| L1 | **Fragile `replace('</div>', '')`** — stripped ALL `</div>` occurrences in `cloud_html`. Any nested `<div>` inside the cloud block would produce malformed HTML silently. | `build_page()` L~1156 | `_topic_cloud()` now returns **two independent fragment strings** (`sidebar_html`, `inline_chips`) instead of one string that requires post-processing. Both are generated from the same data in one pass. |
| L2 | **Unbounded `fact_confidence`** — `int(art.get("fact_confidence", 3))` with no clamp. Values outside 0–5 from a bad JSON would render 0 or 8+ dots, breaking the visual. | `_article_card()` | Clamped: `conf = max(0, min(5, int(...)))`. |
| L3 | **Trailing space in class attr** — `f'<div class="notion-section {extra_cls}">'` with `extra_cls=""` produced `class="notion-section "` (trailing space). CSS specificity works, but it's technically invalid. | `_notion_section()` | Changed to `f'notion-section {extra_cls}'.rstrip()` for the class string. |
| L4 | **`resolve_date()` fallback was misleading** — fell back to `_ist_today()` when no data found, then immediately failed with "no articles for today". Two confusing errors instead of one clear one. | `resolve_date()` | Now returns `None` explicitly when no data found in either `notes/` or `filtered/`. Caller emits one diagnostic error message with next-step guidance. |

---

### P2 — Code quality

| # | Finding | Fix applied |
|---|---------|-------------|
| Q1 | `import glob` — imported, never used (code uses `Path.glob()`). | Removed. |
| Q2 | `from __future__ import annotations` — unnecessary; no forward-reference strings used, Python 3.9+ supports `list[dict]` natively. | Removed. |
| Q3 | `conf_dot` class had extra space: `{"filled" if …}` used `"  filled"` with two spaces by mistake. | Fixed to `" filled"`. |
| Q4 | No `--dry-run` flag — made it impossible to test the build step in CI without writing to disk. | Added `--dry-run` to CLI and to the workflow `inputs`. |
| Q5 | No `--out-file` option — required callers to infer the output path from `--out-dir` + date. The workflow couldn't target `docs/index.html` directly. | Added `--out-file` (takes precedence over `--out-dir`). |

---

### P3 — Deployment

| # | Finding | Fix applied |
|---|---------|-------------|
| D1 | **Output path mismatch** — module wrote to `output/web/YYYY-MM-DD/index.html` but GitHub Pages needs `docs/index.html`. | Workflow passes `--out-file docs/index.html` explicitly. Module default path still works for local use. |
| D2 | **Google Fonts CDN** — live network request on every page load. If CDN is unreachable (offline, firewall), fonts silently fall back to system fonts — acceptable, but no `font-display` hint was present. | Added `<link rel="preconnect">` and `display=swap` in the Fonts URL so text renders immediately with system fonts while custom fonts load. |
| D3 | **Workflow missing `workflow_call` inputs/outputs** — needed to allow `notes_writer.yml` to chain into this job and pass the built date downstream. | Added `outputs` block (`deployed_date`, `output_size_kb`) to the workflow. |
| D4 | **No concurrency guard** — two simultaneous builds could race on `docs/index.html` and cause a dirty commit. | Added `concurrency: group: web-builder-${{ github.ref_name }}`. |
| D5 | **Date not re-validated in workflow shell** — a bad `--date` value would reach Python before being caught, producing a confusing traceback rather than a clear shell-level error. | Workflow `resolve_date` step validates format with `grep -Eq` before passing to Python. Defence-in-depth on top of Python-level `_validate_date()`. |

---

## Dependency Analysis

| Layer | Dependency | Type | Risk |
|---|---|---|---|
| Python runtime | stdlib only (`argparse`, `html`, `json`, `re`, `sys`, `datetime`, `pathlib`) | Zero-install | None |
| Fonts (runtime, browser) | `fonts.googleapis.com` | CDN, external | Low — graceful fallback via `font-display:swap` + system font stack |
| CI runner | `ubuntu-latest`, `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`, `actions/github-script@v7` | GitHub Actions | Pinned by GitHub; standard |

**pip dependencies required: zero.**

---

## How to wire into `notes_writer.yml`

Add this block at the end of `notes_writer.yml` to auto-trigger web build after notes are committed:

```yaml
  build_web:
    name: "Build web (call web_builder)"
    needs: write_notes
    if: success()
    uses: ./.github/workflows/web_builder.yml
    with:
      date: ${{ needs.write_notes.outputs.target_date }}
```

`write_notes` job in `notes_writer.yml` must expose `target_date` as an output:

```yaml
    outputs:
      target_date: ${{ steps.run_notes.outputs.target_date }}
```
