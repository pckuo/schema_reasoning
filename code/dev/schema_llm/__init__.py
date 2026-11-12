"""schema_llm: generate latent event-schema stimuli for LLM probing.

Pipeline (per scenario): sample a recipe -> build a generator prompt -> an LLM
realizes it into natural text + emits scene labels -> a verifier checks the
forbidden-lexicon constraint and aligns spans. See main/plans/SCHEMA_PROBING_PLAN.md.
"""
