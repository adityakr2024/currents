"""
notes_core/tiers.py
====================
8 generation tiers — complete and final.

Tier is determined AFTER all engines have run for an article.
Inputs: what succeeded, what failed, what was disabled.

TIER DEFINITIONS
─────────────────
  llm_grounded_bilingual    LLM + Grounding + Hindi translation
  llm_grounded_en_only      LLM + Grounding, Hindi failed/disabled
  llm_ungrounded_bilingual  LLM + Hindi, Grounding failed/disabled
  llm_ungrounded_en_only    LLM only, Grounding + Hindi both failed/disabled
  grounded_extractive       Grounding + Sumy, LLM down, has article text
  offline_extractive        Sumy only, everything else down/disabled
  grounded_snippets_only    Grounding worked, but no article text and LLM down
  title_only_record         Everything failed — only title + URL preserved

NEEDS RETRY LOGIC
──────────────────
An article goes into needs_retry_*.csv when:
  - tier is title_only_record  (total failure)
  - tier is grounded_snippets_only (LLM was down mid-run)
  - tier is offline_extractive AND llm_exhausted_at_run=True (skipped, not unavailable)
  - translation_method is all_failed (notes written but Hindi completely missing)
"""

from __future__ import annotations

# ── Tier constants ─────────────────────────────────────────────────────────────
LLM_GROUNDED_BILINGUAL   = "llm_grounded_bilingual"
LLM_GROUNDED_EN_ONLY     = "llm_grounded_en_only"
LLM_UNGROUNDED_BILINGUAL = "llm_ungrounded_bilingual"
LLM_UNGROUNDED_EN_ONLY   = "llm_ungrounded_en_only"
GROUNDED_EXTRACTIVE      = "grounded_extractive"
OFFLINE_EXTRACTIVE       = "offline_extractive"
GROUNDED_SNIPPETS_ONLY   = "grounded_snippets_only"
TITLE_ONLY_RECORD        = "title_only_record"

ALL_TIERS = [
    LLM_GROUNDED_BILINGUAL,
    LLM_GROUNDED_EN_ONLY,
    LLM_UNGROUNDED_BILINGUAL,
    LLM_UNGROUNDED_EN_ONLY,
    GROUNDED_EXTRACTIVE,
    OFFLINE_EXTRACTIVE,
    GROUNDED_SNIPPETS_ONLY,
    TITLE_ONLY_RECORD,
]

# Translation method constants
TRANS_INDICTRANS2  = "indictrans2"
TRANS_BHASHINI     = "bhashini"
TRANS_LIBRETRANSLATE = "libretranslate"
TRANS_LLM_FALLBACK = "llm_fallback"
TRANS_ALL_FAILED   = "all_failed"
TRANS_DISABLED     = "disabled"
TRANS_NOT_APPLICABLE = "not_applicable"


def decide_tier(
    llm_ok: bool,
    grounding_ok: bool,
    hindi_ok: bool,
    has_text: bool,
) -> str:
    """
    Decide generation tier from engine results.

    Args:
        llm_ok:       English notes were generated successfully
        grounding_ok: Tavily/Serper returned results
        hindi_ok:     Hindi translation succeeded
        has_text:     article has full_text or summary (not title_only)
    """
    if llm_ok:
        if grounding_ok and hindi_ok:
            return LLM_GROUNDED_BILINGUAL
        if grounding_ok and not hindi_ok:
            return LLM_GROUNDED_EN_ONLY
        if not grounding_ok and hindi_ok:
            return LLM_UNGROUNDED_BILINGUAL
        return LLM_UNGROUNDED_EN_ONLY

    # LLM not available
    if grounding_ok and has_text:
        return GROUNDED_EXTRACTIVE
    if grounding_ok and not has_text:
        return GROUNDED_SNIPPETS_ONLY
    if has_text:
        return OFFLINE_EXTRACTIVE
    return TITLE_ONLY_RECORD


def needs_retry(tier: str, translation_method: str, llm_exhausted_at_run: bool) -> tuple[bool, str]:
    """
    Decide if an article should go into needs_retry_*.csv.

    Returns:
        (should_retry, reason)
    """
    if tier == TITLE_ONLY_RECORD:
        return True, "total_failure_no_content_no_llm"

    if tier == GROUNDED_SNIPPETS_ONLY:
        return True, "llm_unavailable_no_article_text"

    if tier == OFFLINE_EXTRACTIVE and llm_exhausted_at_run:
        return True, "llm_exhausted_during_run"

    if translation_method == TRANS_ALL_FAILED:
        return True, "hindi_translation_all_providers_failed"

    return False, ""
