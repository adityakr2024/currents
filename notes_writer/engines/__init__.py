"""
engines/ — four independently runnable processing engines.

Each engine:
  - Can be imported and used programmatically
  - Can be run directly: python notes_writer/engines/<engine>.py --file input.csv
  - Has its own failure mode, independent of other engines

sumy_engine   — offline, no network, always available
ground_engine — Tavily/Serper via AIPOOL
llm_engine    — AIPOOL LLM, EN notes + Hindi journalism prompt
trans_engine  — Bhashini → IndicTrans2 → LibreTranslate → LLM fallback
"""
