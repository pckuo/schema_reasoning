"""Binding-difficulty stimulus generation (experiments 1A / 1B / 1C).

Separate from ``pipeline`` on purpose: ``pipeline`` is role-first (walk the roles, draw a
referent each) and every durable corpus under ``results/schema_llm/data/`` depends on its exact
sampling behaviour. This subpackage is **cast-first** (draw people, assign roles onto them), which
is what makes B / R / D / E independently controllable, lets one person hold several roles, and
lets a role be empty.

See ``main/plans/20260826-BINDING_GENERATION_PLAN.md``.
"""
