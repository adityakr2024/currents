"""
engines/trans_engine.py
========================
Hindi translation — 4-provider fallback chain.
Bhashini (priority 1) → IndicTrans2/HF (priority 2) → LibreTranslate (priority 3) → LLM (priority 4)

Translates field-by-field. If one field fails, others still translate.
Failed fields stay in English with a flag in translation_method.

DIRECTLY RUNNABLE:
  python notes_writer/engines/trans_engine.py --file notes.csv
  python notes_writer/engines/trans_engine.py --file articles.csv --text-col summary
  python notes_writer/engines/trans_engine.py --file notes.csv --no-bhashini --no-indictrans2

Mode 1 (default): reads en_* columns → writes hi_* columns
Mode 2 (--text-col): translates a single named column → writes hi_<col>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time  # kept: used by NMT provider retry logic (see commented classes above)
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
_G, _Y, _R, _RS = "\033[92m", "\033[93m", "\033[91m", "\033[0m"

_EN_FIELDS = [
    "en_why_in_news","en_significance","en_background",
    "en_key_dimensions","en_analysis",
    "en_prelims_facts","en_mains_questions",
]
_EN_TO_SECTION = {f: f[3:] for f in _EN_FIELDS}  # "en_why_in_news" → "why_in_news"


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: _Bhashini, _IndicTrans2, _LibreTranslate are disabled — no translation API keys available.
# Translation is handled exclusively by the LLM fallback (_LLMFallback).
# To re-enable, uncomment these classes and restore the provider loop in from_config.

# class _Bhashini:
#     """Bhashini ULCA — Government of India, IndicTrans2 underneath. Priority 1."""
#
#     def __init__(self, cfg: dict, user_id: str, ulca_key: str, inference_key: str):
#         import requests as _req
#         self._req         = _req
#         self._config_url  = cfg.get("config_endpoint",
#             "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline")
#         self._pipeline_id = cfg.get("pipeline_id", "64392f96daac500b55c543cd")
#         self._src         = cfg.get("source_language", "en")
#         self._tgt         = cfg.get("target_language", "hi")
#         self._timeout     = cfg.get("timeout_seconds", 30)
#         self._retries     = cfg.get("max_retries", 2)
#         self._user_id     = user_id
#         self._ulca_key    = ulca_key
#         self._inf_key     = inference_key
#         self._callback_url: Optional[str] = None
#         self._service_id:   Optional[str] = None
#         self._cb_key:       Optional[str] = None
#
#     def _config(self) -> None:
#         body = {"pipelineTasks":[{"taskType":"translation","config":{"language":{
#             "sourceLanguage":self._src,"targetLanguage":self._tgt}}}],
#             "pipelineRequestConfig":{"pipelineId":self._pipeline_id}}
#         r = self._req.post(self._config_url,
#             headers={"userID":self._user_id,"ulcaApiKey":self._ulca_key},
#             json=body, timeout=self._timeout)
#         r.raise_for_status()
#         data = r.json()
#         self._service_id   = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
#         ep                 = data["pipelineInferenceAPIEndPoint"]
#         self._callback_url = ep["callbackUrl"]
#         self._cb_key       = ep["inferenceApiKey"]["value"]
#
#     def translate(self, text: str) -> str:
#         if not text or not text.strip():
#             return text
#         if not self._callback_url:
#             self._config()
#         body = {"pipelineTasks":[{"taskType":"translation","config":{
#             "language":{"sourceLanguage":self._src,"targetLanguage":self._tgt},
#             "serviceId":self._service_id}}],
#             "inputData":{"input":[{"source":text}],"language":{"sourceLanguage":self._src}}}
#         for attempt in range(1, self._retries + 2):
#             try:
#                 r = self._req.post(self._callback_url,
#                     headers={self._cb_key: self._cb_key},
#                     json=body, timeout=self._timeout)
#                 r.raise_for_status()
#                 return str(r.json()["pipelineResponse"][0]["output"][0]["target"]).strip()
#             except Exception as exc:
#                 if attempt <= self._retries:
#                     time.sleep(2); continue
#                 raise
#         raise RuntimeError("Bhashini: retries exhausted")


# class _IndicTrans2:
#     """HuggingFace Inference API — IndicTrans2. Priority 2."""
#
#     def __init__(self, cfg: dict, api_key: str):
#         import requests as _req
#         self._req     = _req
#         model         = cfg.get("model","ai4bharat/indictrans2-en-indic-1B")
#         endpoint      = cfg.get("endpoint","https://router.huggingface.co/hf-inference/models/{model}")
#         self._url     = endpoint.replace("{model}", model)
#         self._src     = cfg.get("src_lang","eng_Latn")
#         self._tgt     = cfg.get("tgt_lang","hin_Deva")
#         self._timeout = cfg.get("timeout_seconds", 30)
#         self._retries = cfg.get("max_retries", 2)
#         self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#
#     def translate(self, text: str) -> str:
#         if not text or not text.strip():
#             return text
#         payload = {"inputs": text, "parameters": {"src_lang": self._src, "tgt_lang": self._tgt}}
#         for attempt in range(1, self._retries + 2):
#             try:
#                 r = self._req.post(self._url, headers=self._headers,
#                                    json=payload, timeout=self._timeout)
#                 if r.status_code in (503, 429):
#                     wait = 10 if r.status_code == 503 else 5
#                     if attempt <= self._retries:
#                         time.sleep(wait); continue
#                 r.raise_for_status()
#                 data = r.json()
#                 if isinstance(data, list) and data:
#                     return str(data[0].get("translation_text","")).strip()
#                 return str(data.get("translation_text", data.get("generated_text",""))).strip()
#             except Exception as exc:
#                 if attempt <= self._retries:
#                     time.sleep(2); continue
#                 raise
#         raise RuntimeError("IndicTrans2: retries exhausted")


# class _LibreTranslate:
#     """LibreTranslate — open source. Priority 3."""
#
#     def __init__(self, cfg: dict, api_key: str = ""):
#         import requests as _req
#         self._req     = _req
#         self._url     = cfg.get("endpoint","https://libretranslate.com/translate")
#         self._src     = cfg.get("source","en")
#         self._tgt     = cfg.get("target","hi")
#         self._timeout = cfg.get("timeout_seconds", 20)
#         self._retries = cfg.get("max_retries", 1)
#         self._key     = api_key
#
#     def translate(self, text: str) -> str:
#         if not text or not text.strip():
#             return text
#         payload = {"q": text, "source": self._src, "target": self._tgt, "format": "text"}
#         if self._key:
#             payload["api_key"] = self._key
#         for attempt in range(1, self._retries + 2):
#             try:
#                 r = self._req.post(self._url, json=payload, timeout=self._timeout)
#                 r.raise_for_status()
#                 return str(r.json().get("translatedText","")).strip()
#             except Exception as exc:
#                 if attempt <= self._retries:
#                     time.sleep(2); continue
#                 raise
#         raise RuntimeError("LibreTranslate: retries exhausted")


class _LLMFallback:
    """AIPOOL LLM — journalism prompt. Priority 4."""

    def __init__(self, cfg: dict, pool):
        self._pool        = pool
        self._max_tokens  = cfg.get("max_tokens", 2000)
        self._max_attempts= cfg.get("max_attempts", 2)

    def translate_notes(self, english_notes: dict, article: dict,
                         mq: int, pf: int, kd: int) -> dict:
        from engines.llm_engine  import build_hindi_prompt, call as llm_call, LLMCallError
        from notes_core.parser   import parse_notes
        sys_hi, usr_hi = build_hindi_prompt(article, english_notes, mq=mq, pf=pf, kd=kd)
        raw = llm_call(self._pool, sys_hi, usr_hi, max_tokens=self._max_tokens,
                       max_attempts=self._max_attempts, label="HI-LLM-fallback")
        return parse_notes(raw)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TranslationEngine:
    def __init__(self, providers: list, llm_fallback: Optional[_LLMFallback]):
        self._providers   = providers   # [(name, provider_obj), ...]
        self._llm_fallback = llm_fallback

    @classmethod
    def from_config(cls, cfg: dict, pool, secrets: dict) -> "TranslationEngine":
        # NOTE: Dedicated translation APIs (Bhashini, IndicTrans2, LibreTranslate) are disabled.
        # Only the LLM fallback is used for translation until API keys are available.
        # To re-enable providers, uncomment the three provider classes above and restore
        # the provider loop here.
        tc = cfg.get("translation", {})
        providers = []  # No NMT providers — LLM fallback only.

        # Uncomment the block below (and the provider classes above) to re-enable NMT providers:
        # order = sorted(
        #     [k for k in tc if k != "llm_fallback" and tc[k].get("enabled", True)],
        #     key=lambda k: tc[k].get("priority", 99),
        # )
        # for name in order:
        #     pcfg = tc[name]
        #     try:
        #         if name == "bhashini":
        #             uid  = secrets.get("BHASHINI_USER_ID","")
        #             ukey = secrets.get("BHASHINI_ULCA_API_KEY","")
        #             ikey = secrets.get("BHASHINI_INFERENCE_KEY","")
        #             if not (uid and ukey and ikey):
        #                 log.warning("TransEngine: bhashini skipped — missing BHASHINI_* secrets")
        #                 continue
        #             providers.append((name, _Bhashini(pcfg, uid, ukey, ikey)))
        #         elif name == "indictrans2":
        #             key = secrets.get("HF_API_KEY","")
        #             if not key:
        #                 log.warning("TransEngine: indictrans2 skipped — HF_API_KEY not set")
        #                 continue
        #             providers.append((name, _IndicTrans2(pcfg, key)))
        #         elif name == "libretranslate":
        #             providers.append((name, _LibreTranslate(pcfg, secrets.get("LIBRETRANSLATE_API_KEY",""))))
        #     except Exception as exc:
        #         log.warning("TransEngine: failed to init %s: %s", name, exc)

        log.info("TransEngine: dedicated translation APIs disabled — using LLM fallback only")

        llm_fb = None
        llm_cfg = tc.get("llm_fallback", {})
        if llm_cfg.get("enabled", True) and pool:
            llm_fb = _LLMFallback(llm_cfg, pool)

        active = [n for n, _ in providers] + (["llm_fallback"] if llm_fb else [])
        log.info("TransEngine chain: %s", " -> ".join(active) if active else "NONE")
        return cls(providers, llm_fb)

    def _translate_one(self, text: str, label: str = "") -> tuple[str, str]:
        """Try providers in order. Returns (translated, provider_name)."""
        if not text or not text.strip():
            return text, "passthrough"
        for name, provider in self._providers:
            try:
                result = provider.translate(text)
                if result and result.strip() and result != text:
                    return result, name
            except Exception as exc:
                log.warning("TransEngine: %s failed for '%s': %s", name, label[:30], exc)
        return text, "nmt_failed"

    def _translate_list(self, items: list[str], label: str) -> tuple[list[str], str]:
        if not items:
            return items, "passthrough"
        results, used = [], "passthrough"
        for i, item in enumerate(items):
            t, m = self._translate_one(item, f"{label}[{i}]")
            results.append(t)
            if m not in ("passthrough","nmt_failed"):
                used = m
        return results, used

    def _translate_dims(self, dims: list[dict], label: str) -> tuple[list[dict], str]:
        if not dims:
            return dims, "passthrough"
        results, used = [], "passthrough"
        for i, dim in enumerate(dims):
            th, mh = self._translate_one(dim.get("heading",""), f"{label}[{i}].heading")
            tc, mc = self._translate_one(dim.get("content",""), f"{label}[{i}].content")
            results.append({"heading": th, "content": tc})
            for m in (mh, mc):
                if m not in ("passthrough","nmt_failed"):
                    used = m
        return results, used

    def translate_notes(
        self,
        english_notes: dict,
        article: dict,
        mq: int = 2, pf: int = 4, kd: int = 4,
    ) -> tuple[dict, str]:
        """Translate full notes dict. Returns (hindi_notes, method)."""
        if not self._providers:
            if self._llm_fallback:
                try:
                    result = self._llm_fallback.translate_notes(english_notes, article, mq, pf, kd)
                    return result, "llm_fallback"
                except Exception as exc:
                    log.error("TransEngine: LLM fallback failed: %s", exc)
            return {}, "all_failed"

        hi: dict = {}
        methods_used = []

        for field in ("why_in_news","significance","background","analysis"):
            val = english_notes.get(field,"")
            t, m = self._translate_one(val, field)
            hi[field] = t
            methods_used.append(m)

        for field in ("prelims_facts","mains_questions"):
            val = english_notes.get(field,[])
            t, m = self._translate_list(val, field)
            hi[field] = t
            methods_used.append(m)

        dims = english_notes.get("key_dimensions",[])
        t, m = self._translate_dims(dims, "key_dimensions")
        hi["key_dimensions"] = t
        methods_used.append(m)

        real_methods = [m for m in methods_used if m not in ("passthrough","nmt_failed","")]
        if real_methods:
            from collections import Counter
            overall = Counter(real_methods).most_common(1)[0][0]
        else:
            overall = "nmt_failed"

        if overall == "nmt_failed" and self._llm_fallback:
            try:
                log.info("TransEngine: NMT all failed — trying LLM fallback")
                result = self._llm_fallback.translate_notes(english_notes, article, mq, pf, kd)
                return result, "llm_fallback"
            except Exception as exc:
                log.error("TransEngine: LLM fallback failed: %s", exc)
                return hi, "all_failed"

        return hi, overall


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _output_path(input_path: Path, output_dir: Optional[str], ext: str) -> Path:
    name = f"{input_path.stem}_translated{ext}"
    if output_dir:
        return Path(output_dir) / name
    return input_path.parent / name


def _pipe_to_list(val: str) -> list[str]:
    if not val or not val.strip():
        return []
    return [v.strip() for v in str(val).split("|") if v.strip()]


def _run_standalone(args: argparse.Namespace) -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )

    _HERE = Path(__file__).resolve().parent.parent
    _REPO = _HERE.parent
    for _p in [str(_HERE), str(_REPO)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        from AIPOOL import PoolManager
    except ImportError:
        print("ERROR: AIPOOL not found. Run from repo root.", file=sys.stderr)
        return 1

    from notes_core.loader import load
    import yaml

    input_path = Path(args.file)
    articles   = load(input_path)
    log.info("Loaded %d rows", len(articles))

    # Build config with CLI overrides
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "notes_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    tc  = cfg.setdefault("translation", {})
    if args.no_bhashini:       tc.setdefault("bhashini",{})["enabled"]       = False
    if args.no_indictrans2:    tc.setdefault("indictrans2",{})["enabled"]     = False
    if args.no_libretranslate: tc.setdefault("libretranslate",{})["enabled"]  = False
    if args.no_llm_translate:  tc.setdefault("llm_fallback",{})["enabled"]    = False

    secrets = {
        "HF_API_KEY":             os.getenv("HF_API_KEY",""),
        "BHASHINI_USER_ID":       os.getenv("BHASHINI_USER_ID",""),
        "BHASHINI_ULCA_API_KEY":  os.getenv("BHASHINI_ULCA_API_KEY",""),
        "BHASHINI_INFERENCE_KEY": os.getenv("BHASHINI_INFERENCE_KEY",""),
        "LIBRETRANSLATE_API_KEY": os.getenv("LIBRETRANSLATE_API_KEY",""),
    }
    pool   = PoolManager.from_config(module="trans_engine_standalone")
    engine = TranslationEngine.from_config(cfg, pool=pool, secrets=secrets)
    rows   = []

    for article in articles:
        raw = article.get("_raw", {})
        row = {k: v for k, v in raw.items()}
        row["url"] = article.get("url","")   # mandatory

        if args.text_col:
            # Mode 2 — translate single column.
            # NOTE: LLM fallback uses a structured UPSC notes prompt and cannot translate
            # arbitrary columns. With NMT providers disabled, Mode 2 cannot translate;
            # the column is copied as-is with method "nmt_unavailable".
            if not engine._providers:
                log.warning(
                    "Mode 2 (--text-col) requires an NMT provider. "
                    "LLM fallback only supports structured notes (Mode 1). "
                    "Column '%s' will not be translated.", args.text_col
                )
                row[f"hi_{args.text_col}"] = raw.get(args.text_col, "")
                row["translation_method"]  = "nmt_unavailable"
            else:
                text    = raw.get(args.text_col,"")
                hi_text, method = engine._translate_one(text, args.text_col)
                row[f"hi_{args.text_col}"] = hi_text
                row["translation_method"]  = method
        else:
            # Mode 1 — translate en_* notes columns
            en_notes: dict = {}
            for en_field, section_key in _EN_TO_SECTION.items():
                val = article.get(en_field,"") or raw.get(en_field,"")
                if not val:
                    continue
                if section_key == "key_dimensions":
                    dims = []
                    for part in str(val).split(" | "):
                        if ":" in part:
                            h, c = part.split(":",1)
                            dims.append({"heading":h.strip(),"content":c.strip()})
                        elif part.strip():
                            dims.append({"heading":"","content":part.strip()})
                    en_notes[section_key] = dims
                elif section_key in ("prelims_facts","mains_questions"):
                    en_notes[section_key] = _pipe_to_list(str(val))
                else:
                    en_notes[section_key] = str(val).strip()

            if not any(en_notes.values()):
                log.warning("No en_* columns found: %s — skipped", article["title"][:50])
                row["translation_method"] = "skipped_no_en_notes"
            else:
                hi_notes, method = engine.translate_notes(en_notes, article)
                for sk, sv in hi_notes.items():
                    col = f"hi_{sk}"
                    if isinstance(sv, list):
                        row[col] = " | ".join(
                            f"{d.get('heading','')}: {d.get('content','')}" if isinstance(d,dict)
                            else str(d) for d in sv)
                    else:
                        row[col] = str(sv) if sv else ""
                row["translation_method"] = method

        rows.append(row)
        log.info("[%s] %s", row.get("translation_method","?"), article["title"][:60])

    if not args.no_csv and rows:
        out = _output_path(input_path, args.output_dir, ".csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               extrasaction="ignore", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, out)
        log.info("CSV → %s", out)

    if not args.no_json and rows:
        out_j = _output_path(input_path, args.output_dir, ".json")
        out_j.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("JSON → %s", out_j)

    methods = {}
    for r in rows:
        m = r.get("translation_method","")
        methods[m] = methods.get(m,0) + 1
    log.info("Done — methods: %s", methods)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Translate en_* notes columns → hi_* (or any text column)")
    p.add_argument("--file",               required=True)
    p.add_argument("--output-dir",         help="Output directory (default: same as input)")
    p.add_argument("--text-col",           help="Mode 2: translate a single named column")
    p.add_argument("--no-bhashini",        action="store_true")
    p.add_argument("--no-indictrans2",     action="store_true")
    p.add_argument("--no-libretranslate",  action="store_true")
    p.add_argument("--no-llm-translate",   action="store_true")
    p.add_argument("--no-csv",             action="store_true")
    p.add_argument("--no-json",            action="store_true")
    p.add_argument("--verbose",            action="store_true")
    sys.exit(_run_standalone(p.parse_args()))
