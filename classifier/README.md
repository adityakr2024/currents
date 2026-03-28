# classifier

Standalone article grader for competitive exam current affairs pipelines.

Reads `articles_*.csv` or `articles_*.json` files produced by the RSS fetcher,
grades every article on a 0–100 scale against the exam syllabus, and writes
enriched output files back to the same dated folder.

---

## How it works

Articles pass through four sequential stages:

```
articles_*.csv / .json
        │
        ▼
  [Gate 1 — Hard Exclude]     Blocklist regex → EXCLUDED (score=0)
        │ pass
        ▼
  [Gate 2 — Base Scorer]      Keywords + event bonuses + penalties → base_score
        │
        ▼
  [Gate 3 — Booster]          High-yield topic boosters → boost_score
        │
        ▼
  [Tagger]                    final_score = base + boost, grade, GS paper, note
        │
        ▼
  classified_*.csv + .json
  needs_fetch_*.csv
```

### Grade bands (configurable in `config/gates.yaml`)

| Score  | Grade    | Meaning                              |
|--------|----------|--------------------------------------|
| 0–20   | EXCLUDED | Noise — sports, entertainment, etc.  |
| 21–45  | LOW      | Borderline / tangentially relevant   |
| 46–70  | MEDIUM   | Relevant — worth tracking            |
| 71–100 | HIGH     | Core syllabus — must-read            |

---

## Usage

```bash
# Install dependency
pip install PyYAML

# Today's latest file (auto-discovered from ../data/)
python classify.py

# Specific date folder
python classify.py --date 2026-03-18

# Specific file
python classify.py --file ../data/2026-03-20/articles_11-50-03.csv

# Custom data root
python classify.py --data-dir /path/to/data/folder

# Verbose / debug logging
python classify.py --verbose
```

---

## Output files

All output files are written to the **same dated folder** as the input file.

| File                         | Contents                                    |
|------------------------------|---------------------------------------------|
| `classified_HH-MM-SS.csv`   | All articles + classifier columns           |
| `classified_HH-MM-SS.json`  | Same data as structured JSON                |
| `needs_fetch_HH-MM-SS.csv`  | Articles with no `article_text` (fetch later) |

### New columns in classified output

| Column               | Description                                      |
|----------------------|--------------------------------------------------|
| `gate`               | `EXCLUDED` / `LOW` / `MEDIUM` / `HIGH`           |
| `final_score`        | 0–100                                            |
| `base_score`         | Score before booster                             |
| `boost_score`        | Booster contribution                             |
| `gs_paper`           | `GS1` / `GS2` / `GS3` / `GS4`                   |
| `topic_label`        | Primary syllabus topic                           |
| `matched_topics`     | Comma-separated matched topic names              |
| `classification_note`| Human-readable scoring summary                   |
| `text_present`       | `true` / `false`                                 |

Original columns are preserved unchanged.

---

## Input auto-detection

The loader accepts any CSV or JSON with these column names (case-insensitive):

| Field         | Accepted names                                          |
|---------------|---------------------------------------------------------|
| Title         | `title`, `headline`, `head`, `article_title`            |
| URL           | `url`, `link`, `article_url`, `source_url`              |
| Summary       | `summary`, `description`, `excerpt`, `preview`          |
| Article text  | `article_text`, `text`, `full_text`, `body`, `content`  |
| Source        | `source`, `publisher`, `feed_source`                    |
| Published     | `published`, `date`, `published_at`, `pub_date`         |

Unknown columns are passed through to output unchanged.

---

## Configuration

All tunable parameters live in `config/` — no code changes needed for tuning.

| File               | Controls                                                |
|--------------------|---------------------------------------------------------|
| `config/gates.yaml`| Score band thresholds, bonuses, penalties, output opts  |
| `config/topics.yaml`| Blocklist patterns, event bonuses, topic keywords, GS  |
|                    | mapping, boosters, anchor terms, institutions           |

---

## Repo position

```
repo/
├── rss_fetcher/        ← upstream: produces articles_*.csv
├── classifier/         ← this module
│   ├── classify.py
│   ├── requirements.txt
│   ├── config/
│   │   ├── gates.yaml
│   │   └── topics.yaml
│   └── core/
│       ├── loader.py
│       ├── excluder.py
│       ├── scorer.py
│       ├── booster.py
│       ├── tagger.py
│       └── writer.py
└── data/
    └── YYYY-MM-DD/
        ├── articles_HH-MM-SS.csv      ← input
        ├── classified_HH-MM-SS.csv    ← output
        ├── classified_HH-MM-SS.json   ← output
        └── needs_fetch_HH-MM-SS.csv   ← output
```

---

## Importable API

```python
from classifier.classify import run_pipeline
from pathlib import Path

result = run_pipeline(
    file_path=Path("data/2026-03-20/articles_11-50-03.csv"),
    verbose=True,
)
# result = {
#   "input_file": "...",
#   "output_files": {"csv": "...", "json": "...", "needs_fetch": "..."},
#   "stats": {"total_input": 120, "gate_counts": {...}, ...}
# }
```
