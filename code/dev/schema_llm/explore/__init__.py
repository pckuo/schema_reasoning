"""explore: iteration-first analysis sandbox (see README.md).

Version-controlled + importable (reuse the package toolkit: latents, _stats, _decode, ...), but with
a lighter authoring contract than the main subpackages: ruff + types + inline asserts/controls are
always on; formal unit tests + parity are deferred to graduation. Keep it a LEAF — no other
schema_llm module may import from explore/, so explorations may import anything cycle-free.
"""
