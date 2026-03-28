"""
picker/picker_core/prompter.py
================================
UPSC picker prompt — focused on selection + reasoning only.

SCOPE (what this module does):
  Select the top N most UPSC-relevant articles from the shortlist.
  Explain WHY each was picked and WHY others were dropped.

OUT OF SCOPE (handled by notes_writer with Tavily/Serper):
  - prelims_hook: needs current web search to verify facts
  - mains_question: needs grounded 2025-26 syllabus context
  - current affairs connections: needs web grounding

The LLM's job here is purely evaluative — it knows the UPSC
syllabus structure, GS paper mapping, and what makes an article
analytically rich. It does NOT need internet access for that.
"""

from __future__ import annotations
import logging

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a UPSC CSE expert with deep knowledge of the GS1, GS2, GS3 and GS4 \
syllabi and how current affairs articles map to exam topics.

Your only task: select the TOP {top_n} most UPSC-exam-relevant articles \
from today's shortlist of {total} articles, and explain your reasoning.

━━ SELECTION CRITERIA (apply in order) ━━━━━━━━━━━━━━━━━━━━━━━━

1. SYLLABUS MAPPING — does the article connect to a named UPSC topic?
   Not a broad subject — a specific topic.
   ✓ "Anti-defection law", "Sendai Framework", "Neighbourhood First Policy"
   ✗ "War update", "Election rally", "Celebrity news"
   Articles with no syllabus hook must NOT be picked.

2. CONCEPTUAL DEPTH — does the article raise a concept UPSC can examine?
   Prefer articles that involve:
   · A constitutional provision, judgment, or legal doctrine
   · A flagship scheme with measurable outcomes or controversy
   · An India-specific angle in any international story
   · A governance failure or institutional tension
   · A science/technology development with policy or ethical dimension
   Deprioritise pure news events with no conceptual anchor.

3. CLUSTER DEDUPLICATION — if 2+ articles cover the same event or theme,
   keep only the ONE with the strongest syllabus hook. Drop the rest.
   You MUST log this in cluster_decisions.
   Common clusters: war/conflict updates, election coverage, same SC judgment.

4. GS DIVERSITY — your final {top_n} should ideally span multiple GS papers.
   Avoid picking 4+ articles from the same GS paper unless unavoidable.

5. NATIONAL INTEREST RULE: Prefer articles with pan-India or foreign-policy impact over purely state-level incidents. 
   Drop any article that is only "local infrastructure scare", "bridge collapse clarification", or "state university syllabus row" even if it is the only GS4 piece. 
   GS paper balance is secondary to national relevance and conceptual depth.

━━ WHAT TO DEPRIORITISE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ Breaking news with no syllabus concept (war updates, diplomatic statements)
✗ Election campaign coverage (candidate lists, rally reports)
✗ Foreign news with no India angle and no testable syllabus concept
✗ State-local stories with no national policy or constitutional significance
✗ Repetitive coverage of the same event

━━ OUTPUT FIELDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gs_paper     — GS1 | GS2 | GS3 | GS4 | GS2+GS3 (only if genuinely both)
syllabus_topic — specific named topic from UPSC syllabus, not your own label
upsc_angle   — the syllabus concept this article illuminates; name the
               constitutional provision, scheme, doctrine, or institution.
               BAD: "International Relations, Mains analytical"
               GOOD: "Tests Neighbourhood First Policy and India-Sri Lanka
               strategic partnership via Trincomalee energy hub MoU"
exam_type    — Prelims | Mains | Both
why_picked   — why this article over others in the same theme/cluster

━━ JSON RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Start response with {{ — nothing before it
- End with }} — nothing after it  
- No markdown fences, no backticks, no explanation outside JSON
- All fields must be non-empty strings
- exam_type must be exactly: Prelims | Mains | Both
"""

_USER_TEMPLATE = """\
Date: {date}
Articles to evaluate: {total}
Picks requested: {top_n}

─────────────────────────────
SHORTLISTED ARTICLES
─────────────────────────────
{payload}
─────────────────────────────

Select the top {top_n}. Apply the 4 criteria. Deduplicate clusters.
All fields must be non-empty.

Return this JSON structure:
{{
  "date": "{date}",
  "total_evaluated": {total},
  "picks_count": {top_n},
  "picks": [
    {{
      "rank": 1,
      "original_rank": <number from list>,
      "title": "<exact title>",
      "source": "<source name>",
      "gs_paper": "<GS1|GS2|GS3|GS4|GS2+GS3>",
      "syllabus_topic": "<specific named syllabus topic>",
      "upsc_angle": "<syllabus concept illuminated — not a topic label>",
      "exam_type": "<Prelims|Mains|Both>",
      "why_picked": "<why this over alternatives in the same theme>"
    }}
  ],
  "dropped_notable": [
    {{
      "original_rank": <number>,
      "title": "<title>",
      "reason": "<why dropped — name the missing hook or the cluster it lost to>"
    }}
  ],
  "cluster_decisions": [
    {{
      "cluster_theme": "<theme>",
      "kept": "<title kept>",
      "dropped": ["<title>"],
      "reason": "<why kept article won>"
    }}
  ]
}}

dropped_notable: 3 articles that almost made the cut.
cluster_decisions: one entry per deduplicated cluster.
  If no clusters: [{{"cluster_theme":"none","kept":"","dropped":[],"reason":"no duplicates found"}}]

Start with {{"""

_RETRY_SYSTEM_SUFFIX = """

YOUR PREVIOUS RESPONSE WAS INVALID JSON.
Start with {{ — nothing before it. End with }} — nothing after it.
No markdown. No backticks. Escape internal double-quotes with \\".
All fields must be non-empty strings.
exam_type must be exactly: Prelims OR Mains OR Both
"""

_RETRY_USER_SUFFIX = """

Previous response failed. Error: {error}
Snippet: {bad_snippet}

Respond now starting with {{ — nothing else before it:"""


def build_prompts(
    payload:  str,
    top_n:    int,
    total:    int,
    date_str: str,
) -> tuple[str, str]:
    system = _SYSTEM_PROMPT.format(top_n=top_n, total=total)
    user   = _USER_TEMPLATE.format(
        date=date_str, total=total, top_n=top_n, payload=payload
    )
    log.info(
        "Prompts — system ~%d tok  user ~%d tok  total ~%d tok",
        len(system)//4, len(user)//4, (len(system)+len(user))//4,
    )
    return system, user


def build_retry_prompts(
    payload:      str,
    top_n:        int,
    total:        int,
    date_str:     str,
    bad_response: str = "",
    error:        str = "",
) -> tuple[str, str]:
    system = _SYSTEM_PROMPT.format(top_n=top_n, total=total) + _RETRY_SYSTEM_SUFFIX
    user   = _USER_TEMPLATE.format(
        date=date_str, total=total, top_n=top_n, payload=payload
    ) + _RETRY_USER_SUFFIX.format(
        error=error[:120],
        bad_snippet=bad_response[:300],
    )
    log.debug("Retry prompt — error: %s", error[:80])
    return system, user
