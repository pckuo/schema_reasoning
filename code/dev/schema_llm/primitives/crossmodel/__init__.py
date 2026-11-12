"""Cross-model/task analysis on the pooled shared theme space.

Pools every (model, task) depth-merge run's meta-cluster nodes into ONE source-blind theme space
(:mod:`.merge`), then compares reasoning-primitive *usage* (:mod:`.report` / :mod:`.figures`) and
reasoning *flow* (:mod:`.graph` / :mod:`.graph_figures` / :mod:`.graph_report`, plus the
:mod:`.graphstats` adapter) across sources.
See ``main/plans/_done/CROSSMODEL_GRAPH_ANALYSIS_PLAN.md``.

Part of ``primitives``: builds on ``primitives.transition`` fingerprints +
``primitives.behavior_primitives``; the ``explore`` analyses (``primitive_dynamics``) build on it.
"""
