# AIPOOL

Shared LLM API infrastructure for the UPSC GGG pipeline.  
Every pipeline module that needs an LLM call imports from here.

---

## Position in the repo

```
repo/
├── AIPOOL/              ← this module — shared infra, used by all
├── rss_fetcher/
├── classifier/
├── filter/
├── [picker]/            ← imports from AIPOOL
├── [notes_writer]/      ← imports from AIPOOL
├── [quiz_gen]/          ← imports from AIPOOL
└── data/
    └── api_metrics/     ← health_check_*.json + run metrics land here
```

---

## Folder layout

```
AIPOOL/
│
├── __init__.py                  ← PUBLIC: PoolManager, CallResult, AllKeysExhaustedError
├── test_pool.py                 ← health check runner — run this to verify all keys
├── test_pool.yml                ← copy to .github/workflows/ for manual GH Action
├── requirements.txt             ← PyYAML, requests only
├── README.md
├── .gitignore                   ← blocks config/api_keys.yaml from git
│
├── config/
│   ├── api_pool.yaml            ← provider config, models, limits  [COMMITTED]
│   ├── api_keys.yaml.example    ← key template — copy → api_keys.yaml locally [COMMITTED]
│   └── api_keys.yaml            ← your real keys for local dev  [GITIGNORED — NEVER COMMIT]
│
└── core/
    └── pool/
        ├── __init__.py
        ├── models.py            ← dataclasses — secret-safe APIKey, CallResult
        ├── circuit_breaker.py   ← per-key failure tracking, force-trip on auth
        ├── metrics.py           ← in-run display + persistent JSON per day
        ├── key_registry.py      ← discovers keys from env vars + yaml, LRU ordering
        ├── caller.py            ← HTTP calls (openai_compat / gemini / anthropic)
        └── manager.py           ← orchestrator — the one class everything imports
```

---

## Usage from any module

```python
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AIPOOL import PoolManager, AllKeysExhaustedError, CallResult

# Instantiate once per pipeline run
# Auto-discovers AIPOOL/config/api_pool.yaml and AIPOOL/config/api_keys.yaml
pool = PoolManager.from_config(module="picker")

# Make a call — pool handles key selection, failover, metrics automatically
try:
    result = pool.call(
        prompt = "Your prompt here",
        system = "Your system instruction here",
    )
    print(result.content)
    # result.key_id       → which key was used
    # result.provider     → which provider
    # result.model_used   → which model
    # result.tokens_in    → input tokens
    # result.tokens_out   → output tokens
    # result.latency_ms   → call latency

except AllKeysExhaustedError as e:
    print(f"All API keys exhausted: {e}")
    # handle — skip article, log, continue

# Print metrics table in pipeline log
pool.print_metrics_summary()

# Persist metrics to data/api_metrics/YYYY-MM-DD.json
pool.save_metrics(date_str="2026-03-21")
```

---

## Setting up keys

### Local development

```bash
cd AIPOOL
cp config/api_keys.yaml.example config/api_keys.yaml
# Fill in your real keys in config/api_keys.yaml
# This file is gitignored — safe to add real keys here
```

### GitHub Actions (production)

Go to: **Repo → Settings → Secrets and variables → Actions → New repository secret**

```
GROQ_API_1        gsk_your_key_here
GROQ_API_2        gsk_your_second_key
GEMINI_API_1      AIza_your_gemini_key
OPENROUTER_API_1  sk-or-your_openrouter_key
```

Key naming: `{PROVIDER}_API_{NUMBER}` — numeric suffix only.  
Add as many as you have. Pool discovers all of them automatically.

Then in your workflow `.yml`:

```yaml
- name: Run picker
  env:
    GROQ_API_1:       ${{ secrets.GROQ_API_1 }}
    GROQ_API_2:       ${{ secrets.GROQ_API_2 }}
    GEMINI_API_1:     ${{ secrets.GEMINI_API_1 }}
    OPENROUTER_API_1: ${{ secrets.OPENROUTER_API_1 }}
  run: python picker/picker.py --date $TARGET_DATE
```

---

## Health check

Run before any pipeline day to confirm every key is alive:

```bash
# From repo root
python AIPOOL/test_pool.py

# Specific providers only
python AIPOOL/test_pool.py --providers groq gemini

# Show model response in output
python AIPOOL/test_pool.py --verbose

# Custom data output path
python AIPOOL/test_pool.py --data-dir data
```

**Sample output:**

```
═══════════════════════════════════════════════════════════════════════════════
   AIPOOL — HEALTH CHECK   2026-03-21  08:30 UTC
═══════════════════════════════════════════════════════════════════════════════
  Keys to test : 4
  Providers    : gemini, groq
  Prompt       : "Reply with exactly one word: OK"

  KEY                    PROVIDER     MODEL                               LAT   TOKENS  STATUS
  ────────────────────────────────────────────────────────────────────────────────────────────
  [ 1/4] ✓ PASS  GROQ_API_1            groq         llama-3.3-70b-versatile    412ms  tok:8+3  [resp:OK]
  [ 2/4] ✓ PASS  GROQ_API_2            groq         llama-3.3-70b-versatile    388ms  tok:8+3  [resp:OK]
  [ 3/4] ✗ FAIL  GROQ_API_3            groq         llama-3.3-70b-versatile
         ↳ auth: Auth failed (401)
  [ 4/4] ✓ PASS  GEMINI_API_1          gemini       gemini-1.5-flash           521ms  tok:9+1  [resp:OK]

  PROVIDER        PASS  FAIL  AVG_MS
  ────────────────────────────────────────
  groq               2     1     400
  gemini             1     0     521

  OVERALL  3 passed / 1 failed of 4 key(s)

  FAILED KEYS:
    GROQ_API_3             groq         auth: Auth failed (401)

  Metrics saved → data/api_metrics/health_check_2026-03-21.json
```

**Exit codes:** `0` = all pass · `1` = some failed · `2` = no keys found

---

## GitHub Actions health check

1. Copy `AIPOOL/test_pool.yml` → `.github/workflows/test_pool.yml`
2. Add keys as repository secrets
3. **Actions** → **AIPOOL — Health Check** → **Run workflow**

Results committed to `data/api_metrics/health_check_YYYY-MM-DD.json`.

---

## Supported providers

| Priority | Provider   | Primary model                                    | Fallback model                       |
|----------|------------|--------------------------------------------------|--------------------------------------|
| 1        | Groq       | `llama-3.3-70b-versatile`                        | `llama-3.1-8b-instant`               |
| 2        | Gemini     | `gemini-1.5-flash`                               | `gemini-1.5-flash-8b`                |
| 3        | OpenRouter | `meta-llama/llama-3.3-70b-instruct:free`         | `mistralai/mistral-7b-instruct:free` |
| 4        | Cerebras   | `llama3.1-70b`                                   | `llama3.1-8b`                        |
| 5        | Anthropic  | `claude-haiku-4-5-20251001`                      | (same)                               |
| 6        | OpenAI     | `gpt-4o-mini`                                    | `gpt-3.5-turbo`                      |

---

## How key routing works

```
Every pool.call():

  1. Groq keys  — sorted by last_used_at ascending (LRU first)
  2. Other keys — sorted by last_used_at ascending (LRU first)
  3. For each healthy key (circuit not open):
       a. Try primary model
       b. auth (401/403)    → force-trip CB immediately, skip fallback, next key
       c. rate-limit (429)  → skip key, CB NOT tripped, next key
       d. timeout/server    → try fallback model
       e. fallback fails    → record CB failure, next key
  4. All keys exhausted → raise AllKeysExhaustedError

Circuit breaker resets at the start of every fresh pipeline run.
"One key → one call → rotate" = natural rate limiting, no counters needed.
```

---

## Metrics output

| File | Written by | Contents |
|---|---|---|
| `data/api_metrics/YYYY-MM-DD.json` | `pool.save_metrics()` | Per-run API usage per key |
| `data/api_metrics/health_check_YYYY-MM-DD.json` | `test_pool.py` | Per-key health check results |

Both files append across multiple runs per day — never overwrite.  
Only `key_id` stored — **secrets never written to disk**.

---

## Adding a new provider

**1.** Add a block to `AIPOOL/config/api_pool.yaml`:

```yaml
providers:
  myprovider:
    priority: 7
    base_url: "https://api.myprovider.com/v1"
    caller_type: "openai_compat"      # openai_compat | gemini | anthropic
    models:
      primary:  "best-model-name"
      fallback: "fast-model-name"
    timeout_seconds: 30
    max_tokens: 1024
    rate_limit:
      calls_per_minute: 20
      tokens_per_minute: 10000
    env_key_pattern: "MYPROVIDER_API_"
```

**2.** If the provider uses a non-standard API format, add `_call_myprovider()` in `core/pool/caller.py`.

**3.** Add secrets: `MYPROVIDER_API_1=your_key` in GitHub Secrets or local `api_keys.yaml`.

---

## Security summary

| Concern | Protection |
|---|---|
| Keys in GitHub Actions | Repository Secrets — masked in all logs |
| Keys in local dev | `config/api_keys.yaml` — gitignored, never committed |
| Keys in Python logs | `APIKey.__repr__` shows only last 4 chars (`****abcd`) |
| Keys in metrics JSON | Only `key_id` stored (e.g. `GROQ_API_1`), never the secret |
| Keys in error messages | Regex-sanitized before any log or file write |
| YAML injection | `yaml.safe_load()` — blocks `!!python/object` attacks |
| Placeholder keys | Rejected at load time (length + alphanumeric check) |
