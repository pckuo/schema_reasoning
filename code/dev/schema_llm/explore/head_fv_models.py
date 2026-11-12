"""Per-model constants for the head-FV pipeline, so one implementation serves both subjects.

The pipeline was written against Phi-4-mini-reasoning with its constants inlined. Qwen3-4B
differs in every one of them -- width 2,560 not 3,072, 36x32 heads not 32x24, a native chat
template rather than a borrowed instruct one, different EOS ids, and different structural
markers -- so the choice is between parameterising here or forking the modules.

Forking was rejected: two nominally identical pipelines drifting apart silently is a worse
failure than the cost of this indirection, and the whole point of the Qwen run is that its
protocol matches Phi's.

Selection is by ``SCHEMA_HEAD_FV_MODEL`` and defaults to Phi, so every existing call site and
artifact keeps its current behaviour untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_KEY = "phi4mr"
ENV_VAR = "SCHEMA_HEAD_FV_MODEL"


@dataclass(frozen=True)
class ModelSpec:
    """Everything about a subject that the head-FV pipeline needs to know.

    ``geometry`` is ``(blocks, query_heads, kv_heads, hidden_size, head_dim)`` and is asserted
    against the loaded checkpoint rather than trusted, because a mismatched head list slices
    ``o_proj`` out of range and numpy returns an EMPTY block instead of raising.
    """

    key: str
    model: str
    revision: str
    slug: str
    chat_template: str | None
    think: str
    eos_ids: tuple[int, ...]
    structural: tuple[str, ...]
    geometry: tuple[int, int, int, int, int]
    capture_layers: tuple[int, ...]
    merge_source: str
    supports_phi3_cache_fix: bool
    data_root: str

    @property
    def blocks(self) -> int:
        return self.geometry[0]

    @property
    def query_heads(self) -> int:
        return self.geometry[1]

    @property
    def hidden_size(self) -> int:
        return self.geometry[3]

    @property
    def head_count(self) -> int:
        return self.blocks * self.query_heads

    def depth_layer(self, fraction: float) -> int:
        """The block nearest a relative depth, so grids transfer between models.

        Phi injected at L8/L16/L24 of 32 blocks; the finding was about relative depth, not
        absolute index, so Qwen's 36 blocks give L9/L18/L27 for the same .25/.50/.75.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("depth fraction must lie in [0, 1]")
        return round(fraction * self.blocks)


PHI_MINI = ModelSpec(
    key="phi4mr",
    model="microsoft/Phi-4-mini-reasoning",
    revision="0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a",
    slug="phi-4-mini-reasoning",
    # the reasoning checkpoint ships no usable chat template; the pilot borrowed the instruct one
    chat_template="microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083",
    think="auto",
    eos_ids=(200020, 199999),
    structural=(
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
    ),
    geometry=(32, 24, 8, 3072, 128),
    capture_layers=(0, 4, 8, 12, 16, 20, 24, 28),
    merge_source="phi4mr/spot_error",
    # identical short/long LongRoPE factors make the Phi3 preparer's 4k cache reset redundant
    supports_phi3_cache_fix=True,
    data_root="phi_mini_head_fv/v1",
)

QWEN3_4B = ModelSpec(
    key="qwen34b",
    model="Qwen/Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
    slug="Qwen3-4B-think-pre-drgrpo",
    chat_template=None,  # native Qwen3 template
    think="on",
    eos_ids=(151645, 151643),  # <|im_end|>, <|endoftext|>
    # unlike Phi, Qwen's think markers are single vocabulary tokens (151667/151668), so the
    # content mask really removes them; Phi's only ever removed the EOS
    structural=("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"),
    geometry=(36, 32, 8, 2560, 128),
    capture_layers=(0, 4, 8, 12, 16, 20, 24, 28),
    merge_source="qwen34b/spot_error",
    # natively 40,960 positions and no dual rope factors: the Phi3 fix must refuse, not no-op
    supports_phi3_cache_fix=False,
    data_root="qwen3_4b_head_fv/v1",
)

SPECS = {spec.key: spec for spec in (PHI_MINI, QWEN3_4B)}


@dataclass(frozen=True)
class PrimitiveSpec:
    """Where a subject's reused primitives and its AIE selection live.

    Separate from ``ModelSpec`` because these are properties of a *run* rather than of the
    checkpoint: the reader layer is chosen from measured membership support, and the AIE hashes
    pin the exact ranking whose heads the vectors are projected through. Pinning them as
    literals is deliberate -- reading them back from the artifacts at import time would turn the
    provenance check into a tautology.
    """

    key: str
    source: str
    reader_layer: int
    themes: tuple[int, ...]
    n_responses: int
    n_tokens: int
    assigner: str
    aie_root: str
    validation_root: str
    selection_sha256: str
    parent_identity: str
    child_identity: str
    head_source_layers: tuple[int, ...]
    targets: tuple[int, ...]
    target_names: dict
    injection_layers: tuple[int, ...]
    window: int
    degenerate_themes: tuple[int, ...]


PHI_PRIMITIVES = PrimitiveSpec(
    key="phi4mr",
    source="phi4mr/spot_error",
    reader_layer=20,
    themes=(3, 5, 8, 17, 21, 22, 23, 29),
    n_responses=216,
    n_tokens=682037,
    assigner=(
        "results/schema_llm/analysis/primitive_transplant/20260828-single_primitive/"
        "assigner/assigner.npz"
    ),
    aie_root="phi_mini_head_fv/v1/aie/runs/layer_prefix_v2",
    validation_root="phi_mini_head_fv/v1/aie/runs/layer_prefix_v2_validation_v1",
    selection_sha256="a1b4bb0318fe5a3693446c76c770c8b6414dca33a003e7f3fcc9cbf9fe0684c6",
    parent_identity="c8d67d7ec7046e6aede912d35bba58f78e237ef2fa52258d5b58b1ac90d3b4ec",
    child_identity="f67fcc8670b0f4bc30f268978dd7589358adc26b3e259fae26d0e2c27283a2b6",
    head_source_layers=(7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 31),
    targets=(3, 8),
    target_names={3: "Process consistency checking", 8: "Event-role tracking"},
    injection_layers=(8, 16, 24),
    window=10240,
    # measured, not assumed: t22/t23 absorbed 66% of tokens in Phi's collapsed responses
    degenerate_themes=(22, 23),
)

QWEN_PRIMITIVES = PrimitiveSpec(
    key="qwen34b",
    source="qwen34b/spot_error",
    # L16, not Phi's L20 and not the depth-matched L24: each layer carries only 8 of the 35
    # merged themes, and L16 is the ONLY candidate holding both targets (t7 and t3). L20 has no
    # t7, L24 has no t3. The reader is the instrument, so coverage outranks depth parity.
    reader_layer=16,
    themes=(1, 3, 7, 22, 23, 25, 27, 31),
    n_responses=216,
    n_tokens=336930,
    assigner="results/schema_llm/analysis/head_fv_specificity/20260909-qwen-assigner/assigner.npz",
    aie_root="qwen3_4b_head_fv/v1/aie/runs/main",
    validation_root="qwen3_4b_head_fv/v1/aie/runs/main_validation_v1",
    selection_sha256="bf51a57474e37ee07699634c1c27c2b6978286b90d8b79ca32e9bd575181f61d",
    parent_identity="997618dbd5fac1e4fef5ff12a7c3d555ffdd16121193f644d666465eb80e8162",
    child_identity="e943fea741646ed79bb88500eae06b1229eee0e4205b9b4fa64745fb324bc26e",
    head_source_layers=(16, 17, 18, 19, 20, 21, 22, 23, 25, 33),
    targets=(3, 7),
    target_names={3: "Process consistency checking", 7: "Event sequence reconstruction"},
    # depth-matched to Phi's L8/L16/L24 of 32 blocks: .25/.50/.75 of Qwen's 36
    injection_layers=(9, 18, 27),
    window=8192,
    # t22/t23 are both present at Qwen's L16, so Phi's attractor diagnostic transfers; whether
    # Qwen actually collapses into them is a question for the data, not an assumption
    degenerate_themes=(22, 23),
)

PRIMITIVE_SPECS = {spec.key: spec for spec in (PHI_PRIMITIVES, QWEN_PRIMITIVES)}



def spec_for(key: str) -> ModelSpec:
    if key not in SPECS:
        raise ValueError(f"unknown head-FV model {key!r}; known: {sorted(SPECS)}")
    return SPECS[key]


def active() -> ModelSpec:
    """The subject this process is configured for; Phi unless told otherwise."""
    return spec_for(os.environ.get(ENV_VAR, DEFAULT_KEY))


def active_primitives() -> PrimitiveSpec:
    """The reused-primitive and AIE-selection record for the configured subject."""
    key = os.environ.get(ENV_VAR, DEFAULT_KEY)
    if key not in PRIMITIVE_SPECS:
        raise ValueError(f"no primitive spec for {key!r}; known: {sorted(PRIMITIVE_SPECS)}")
    return PRIMITIVE_SPECS[key]
