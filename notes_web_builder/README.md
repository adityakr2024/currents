# notes_web_builder

Converts **The Currents** notes JSON into a self-contained bilingual UPSC website.

**Zero pip dependencies. Stdlib only. Python 3.9+.**

---

## Folder structure

```
notes_web_builder/
├── notes_web_builder.py          ← the entire module (one file)
├── README.md                     ← this file
├── AUDIT.md                      ← security & deployment audit log
└── .github/
    └── workflows/
        └── web_builder.yml       ← GitHub Actions workflow
```

Place this folder at the **repo root** alongside `data/`, `docs/`, and your other modules.

---

## Quick start (local)

```bash
# Build from latest notes (auto-detects date)
python notes_web_builder/notes_web_builder.py

# Specific date
python notes_web_builder/notes_web_builder.py --date 2026-03-28

# Direct output path (for GitHub Pages)
python notes_web_builder/notes_web_builder.py \
  --date 2026-03-28 \
  --data-dir ./data \
  --out-file ./docs/index.html

# Test build without writing anything
python notes_web_builder/notes_web_builder.py --date 2026-03-28 --dry-run
```

---

## Input: data structure

```
data/
├── notes/                             ← PRIMARY source
│   └── 2026-03-28/
│       └── notes_HH-MM-SS.json       ← notes_writer output
└── filtered/                          ← FALLBACK source
    └── 2026-03-28/
        └── toplist_HH-MM-SS.json     ← filter-step output
```

The builder picks the **latest** file by filename (lexicographic sort, reversed).

### Supported JSON formats

**notes_writer format** (`data/notes/…/notes_*.json`):
```json
{
  "meta": { "articles_count": 15, "elapsed_seconds": 90 },
  "notes": [ { ...article... }, ... ]
}
```

**Plain list** (`data/filtered/…/toplist_*.json`):
```json
[ { ...article... }, ... ]
```

### Article fields

| Field | Type | Required | Used for |
|---|---|---|---|
| `title` | string | ✓ | Headline (EN) |
| `title_hi` | string | — | Headline (HI) |
| `why_in_news` | string | — | Pinned callout under title |
| `context` | string | — | Notion section (EN) |
| `context_hi` | string | — | Notion section (HI) |
| `background` | string | — | Notion section (EN) |
| `background_hi` | string | — | Notion section (HI) |
| `key_points` | list[str] | — | Bullet list (EN) |
| `key_points_hi` | list[str] | — | Bullet list (HI) |
| `policy_implication` | string | — | Notion section (EN) |
| `policy_implication_hi` | string | — | Notion section (HI) |
| `gs_paper` | string | — | Badge + left border colour + filter |
| `upsc_topics` | list[str] | — | Topic chips + filter |
| `fact_confidence` | int 0–5 | — | Dot rating |
| `fact_flags` | list[str] | — | Amber verify block |
| `source` | string | — | Footer source label |
| `url` | string | — | Footer link (http/https only; `javascript:` blocked) |
| `published` | string | — | Footer date |
| `q_en` / `a_en` | string | — | Q&A Quick Bites section (EN) |
| `q_hi` / `a_hi` | string | — | Q&A Quick Bites section (HI) |

All fields except `title` are optional — the builder degrades gracefully for missing data.

---

## Wire into main workflow

**Import API** (call from `notes_writer/main.py` or any orchestrator):

```python
from pathlib import Path
from notes_web_builder.notes_web_builder import build_page, load_notes, resolve_date

data_dir = Path("data")
date_str = "2026-03-28"

articles = load_notes(data_dir, date_str)   # returns []  if nothing found
html     = build_page(articles, date_str)   # pure function, no I/O

out = Path("docs/index.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
```

**GitHub Actions** — add to the end of `notes_writer.yml`:

```yaml
  build_web:
    needs: write_notes
    if: success()
    uses: ./.github/workflows/web_builder.yml
    with:
      date: ${{ needs.write_notes.outputs.target_date }}
```

---

## GitHub Pages setup (one-time)

1. Go to **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` (or your default) | Folder: `/docs`
4. Save — your site will be at `https://<user>.github.io/<repo>/`

The workflow commits `docs/index.html` after every build.

---

## Security model

| Vector | Mitigation |
|---|---|
| XSS via article `url` field | `_safe_url()` whitelist — only `http`, `https`, relative paths allowed. `javascript:` and `data:` URIs are replaced with `#`. |
| Path traversal via `--date` | `_validate_date()` enforces strict `YYYY-MM-DD` regex. Checked in Python **and** the workflow shell step. |
| XSS via article text fields | All text rendered through `_e()` (HTML-escape) or `_attr()` (attribute-escape). No `innerHTML` assignments in JS. |
| Injected HTML in filter data-attributes | `_attr()` applied to all `data-topic` and `data-gs` attribute values. |
