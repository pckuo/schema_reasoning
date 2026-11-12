"""CPU-only, newly fitted L16 fine-cluster dictionary; no model loading or replay.

``fit_dictionary`` consumes an ordered roster of ``Capture`` records and a source
manifest with ``dataset_id``, ``data_sha256``, ``runtime_contract_sha256``,
``cohort='discovery'`` and ``attempt_ids``. Each available capture is a content-only
``(sum(content_mask), 3072)`` floating-point .npy array, in consumed-position order.
Its caller-frozen manifest must bind the fields checked in ``_capture_record``.
That evidence is a caller attestation, NOT a GPU replay certificate produced here.

The portable artifact is contract.json, mean.npy, scale.npy, centers.npy,
fit_labels.npy, and manifest.json. The last file is published only after EVERY
saved fit label equals reloaded assignment. Retain its returned dictionary hash
outside this directory: loading/resuming requires that trusted hash, not just a
self-consistent manifest. Loading needs neither source captures nor refitting.

Choices beyond the plan: n_init=4 (the prior fine-k-means default), explicit Lloyd
defaults, one numerical thread, all available rows (no subsampling), float64
scaling/centers/prediction, and chunked direct squared Euclidean distances.
No old labels, temporal/meta merging, PCA, health filters, or residual contrasts.
Fitting is in-memory and needs a separately gated real-data memory budget.
Missing captures remain explicit missing data, never measured zero association.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .head_fv_existing import file_hash
from .head_fv_models import active as active_model
from .head_fv_readouts import ALIGNMENT, N_CLUSTERS, summarize_response
from .head_fv_specificity import MODEL, REVISION, bind_manifest, digest

LAYER = 16
WIDTH = active_model().hidden_size
FORMAT = "head-fv-dictionary-v1"
SCALER_PARAMETERS = {"copy": True, "with_mean": True, "with_std": True}
KMEANS_PARAMETERS = {
    "n_clusters": N_CLUSTERS,
    "init": "k-means++",
    "n_init": 4,
    "max_iter": 300,
    "tol": 1e-4,
    "verbose": 0,
    "random_state": 0,
    "copy_x": True,
    "algorithm": "lloyd",
}
ARRAY_NAMES = ("mean", "scale", "centers", "fit_labels")
COMPONENTS = ("contract.json", *(f"{name}.npy" for name in ARRAY_NAMES))


@dataclass(frozen=True)
class Capture:
    """One attempted response, even when generation/capture failed or was empty.

    ``manifest_sha256`` must be supplied from the caller's immutable capture
    ledger. A matching digest proves binding, not truth of its replay assertions.
    No-file records require ``capture_error`` unless they contain zero content.
    A generation error WITH captured partial content still contributes all rows.
    """

    attempt_id: str
    generation: dict
    activation_path: Path | None = None
    manifest: dict | None = None
    manifest_sha256: str | None = None
    capture_error: str | None = None


def _hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _implementation() -> dict:
    return {
        "code_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "head_fv_dictionary.py", "head_fv_existing.py",
                "head_fv_specificity.py", "head_fv_readouts.py",
            )
        },
        "versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        },
        "scaler": SCALER_PARAMETERS,
        "kmeans": KMEANS_PARAMETERS,
        "pca": None,
        "subsampling": None,
        "health_filter": None,
        "numeric_threads": 1,
        "runtime_dtype": "<f8",
        "fit_labels_dtype": "<i4",
        "prediction": "float64 subtract-then-divide; scipy.cdist sqeuclidean; first argmin",
    }


def _activations(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != WIDTH
        or value.dtype.kind != "f" or value.dtype.itemsize not in (2, 4, 8)
    ):
        raise ValueError(f"activations must be a floating (n, {WIDTH}) array, at most float64")
    for start in range(0, len(value), 1024):
        if not np.isfinite(value[start:start + 1024]).all():
            raise ValueError("activations must be finite")
    return value


def _capture_record(
    capture: Capture, runtime_sha256: str, source_sha256: str | None = None
) -> tuple[dict, np.ndarray | None]:
    if not isinstance(capture.attempt_id, str) or not capture.attempt_id.strip():
        raise ValueError("captures require a nonempty attempt_id")
    generation = capture.generation
    required = ("prompt_ids", "generated_ids", "full_ids", "content_mask")
    if (
        not isinstance(generation, dict)
        or any(not isinstance(generation.get(key), list) for key in required)
        or type(generation.get("n_prompt")) is not int or "stop_reason" not in generation
    ):
        raise ValueError("generation must retain exact IDs, mask, n_prompt, and stop_reason")
    # Reuse the existing exact-ID/EOS/error contract, without inventing assignments.
    summary = summarize_response(
        generation, None, {"validation": 0}, readout_error="not yet assigned"
    )
    count = summary["content_token_count"]
    record = {
        "attempt_id": capture.attempt_id, "generation": generation,
        "content_rows": count, "capture_error": capture.capture_error,
    }
    if capture.capture_error is not None and (
        not isinstance(capture.capture_error, str) or not capture.capture_error.strip()
    ):
        raise ValueError("capture_error must describe the failure")
    if capture.activation_path is None:
        if capture.manifest is not None or capture.manifest_sha256 is not None:
            raise ValueError("capture evidence without an activation file")
        if count and capture.capture_error is None:
            raise ValueError("missing content activations require an explicit capture_error")
        return {
            **record, "status": "capture_error" if capture.capture_error else "empty_content",
            "captured_rows": 0, "missing_content_rows": count,
        }, None
    if capture.capture_error is not None:
        raise ValueError("cannot supply both activations and a capture_error")
    evidence = capture.manifest
    expected_hash = _hash(capture.manifest_sha256, "capture manifest hash")
    if not isinstance(evidence, dict) or digest(evidence) != expected_hash:
        raise ValueError("capture manifest does not match its immutable hash")
    _hash(evidence.get("source_data_sha256"), "capture source data hash")
    _hash(evidence.get("capture_code_sha256"), "capture code hash")
    if source_sha256 is not None and evidence["source_data_sha256"] != source_sha256:
        raise ValueError("capture belongs to a different source dataset")
    path = Path(capture.activation_path)
    activation_hash = file_hash(path)
    if activation_hash != evidence.get("activations_sha256"):
        raise ValueError("activation file does not match its capture manifest")
    array = _activations(np.load(path, mmap_mode="r", allow_pickle=False))
    positions = [generation["n_prompt"] + i
                 for i, content in enumerate(generation["content_mask"]) if content]
    expected = {
        "model": MODEL, "revision": REVISION, "layer": LAYER,
        "activation_site": "block-output-residual",
        "alignment": ALIGNMENT, "clean_replay": True,
        "runtime_contract_sha256": runtime_sha256,
        "full_ids_sha256": digest(generation["full_ids"]),
        "content_mask_sha256": digest(generation["content_mask"]),
        "row_full_positions": positions,
        "shape": [count, WIDTH], "dtype": array.dtype.str,
    }
    if (
        array.shape != (count, WIDTH)
        or digest({key: evidence.get(key) for key in expected}) != digest(expected)
    ):
        raise ValueError("capture lacks matching clean exact-ID/consumed-position evidence")
    return {
        **record, "status": "captured" if count else "empty_content",
        "captured_rows": count, "missing_content_rows": 0,
        "activation_path": str(path), "capture_manifest": evidence,
        "capture_manifest_sha256": expected_hash,
    }, array


def _predict(
    activations: np.ndarray, mean: np.ndarray, scale: np.ndarray, centers: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    x = _activations(activations)
    labels = np.empty(len(x), dtype="<i4")
    for start in range(0, len(x), chunk_size):
        z = np.array(x[start:start + chunk_size], dtype=np.float64, copy=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z -= mean
            z /= scale
            distances = cdist(z, centers, metric="sqeuclidean")
        if not np.isfinite(z).all() or not np.isfinite(distances).all():
            raise ValueError("nonfinite transformed activations or distances")
        labels[start:start + len(z)] = distances.argmin(axis=1)
    return labels


@dataclass(frozen=True)
class FrozenDictionary:
    """Read-only arrays; ``predict`` alone does NOT establish token alignment."""

    dictionary_sha256: str
    contract: dict
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    fit_labels: np.ndarray

    def predict(self, activations: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
        return _predict(activations, self.mean, self.scale, self.centers, chunk_size)


def _read_arrays(directory: Path, n_rows: int) -> dict[str, np.ndarray]:
    arrays = {}
    shapes = {"mean": (WIDTH,), "scale": (WIDTH,),
              "centers": (N_CLUSTERS, WIDTH), "fit_labels": (n_rows,)}
    for name, shape in shapes.items():
        array = np.load(directory / f"{name}.npy", allow_pickle=False)
        dtype = "<i4" if name == "fit_labels" else "<f8"
        if array.shape != shape or array.dtype.str != dtype or not np.isfinite(array).all():
            raise ValueError(f"invalid persisted {name} shape/dtype/values")
        array.setflags(write=False)
        arrays[name] = array
    if (
        np.any(arrays["scale"] <= 0) or np.any(arrays["fit_labels"] < 0)
        or np.any(arrays["fit_labels"] >= N_CLUSTERS)
        or np.unique(arrays["fit_labels"]).size != N_CLUSTERS
    ):
        raise ValueError("invalid persisted scales or fine-cluster IDs")
    return arrays


def load_dictionary(
    directory: Path, *, expected_sha256: str, expected_contract: dict | None = None
) -> FrozenDictionary:
    """Verify every component against an externally retained hash; never refit.

    Code and numerical-library versions must match the frozen contract. Portability
    means ordinary JSON/.npy rather than pickled estimators, not untested numerical
    equivalence across arbitrary software versions.
    """
    expected_sha256 = _hash(expected_sha256, "expected dictionary hash")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    stored_hash = manifest.pop("dictionary_sha256", None)
    if stored_hash != expected_sha256 or digest(manifest) != expected_sha256:
        raise ValueError("dictionary manifest hash mismatch")
    if manifest.get("format") != FORMAT or set(manifest.get("files", {})) != set(COMPONENTS):
        raise ValueError("invalid dictionary component manifest")
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"dictionary component hash mismatch: {name}")
    contract = json.loads((directory / "contract.json").read_text())
    if expected_contract is not None and digest(contract) != digest(expected_contract):
        raise ValueError("refusing changed immutable fit contract")
    if contract["implementation"] != _implementation():
        raise ValueError("dictionary code/versions/parameters differ from this runtime")
    if (contract["model"], contract["revision"], contract["layer"], contract["width"]) != (
        MODEL, REVISION, LAYER, WIDTH
    ):
        raise ValueError("dictionary model/revision/layer/width mismatch")
    rows = contract["accounting"]["fit_rows"]
    certificate = manifest.get("certificate")
    if rows < N_CLUSTERS or certificate != {"fit_rows": rows, "exact_reloaded_labels": rows}:
        raise ValueError("missing exhaustive fit-label certificate")
    return FrozenDictionary(expected_sha256, contract, **_read_arrays(directory, rows))


def fit_dictionary(
    directory: Path, captures: list[Capture], *, source_manifest: dict,
    expected_sha256: str | None = None, chunk_size: int = 1024,
) -> FrozenDictionary:
    """Fit once; identical resume verifies source bytes/contract without rewriting.

    Capture errors are recorded with their missing content counts, not hidden by
    health/correctness filtering. ``capture_complete`` must be reviewed before
    interpreting a real dictionary. No cohort membership or A/B gate is certified.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    source_hash = _hash(source_manifest.get("data_sha256"), "source data hash")
    runtime_hash = _hash(source_manifest.get("runtime_contract_sha256"), "runtime contract hash")
    ids = [capture.attempt_id for capture in captures]
    if (
        not isinstance(source_manifest.get("dataset_id"), str)
        or not source_manifest["dataset_id"].strip()
        or source_manifest.get("cohort") != "discovery"
        or source_manifest.get("attempt_ids") != ids or len(set(ids)) != len(ids)
    ):
        raise ValueError("source manifest must bind the complete ordered discovery attempt roster")
    records, arrays, offset = [], [], 0
    for capture in captures:
        record, array = _capture_record(capture, runtime_hash, source_hash)
        size = record["captured_rows"]
        records.append({**record, "fit_row_range": [offset, offset + size]})
        offset += size
        if array is not None and len(array):
            arrays.append(array)
    missing = sum(record["missing_content_rows"] for record in records)
    contract = {
        "format": FORMAT, "model": MODEL, "revision": REVISION, "layer": LAYER, "width": WIDTH,
        "source": source_manifest, "implementation": _implementation(), "responses": records,
        "accounting": {
            "attempts": len(records), "fit_rows": offset, "missing_content_rows": missing,
            "capture_complete": missing == 0,
            "empty_content_responses": sum(record["content_rows"] == 0 for record in records),
            "capture_errors": sum(record["status"] == "capture_error" for record in records),
            "generation_errors": sum(
                record["generation"]["stop_reason"] == "error" for record in records
            ),
        },
    }
    contract_hash = digest(contract)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        if expected_sha256 is None:
            raise ValueError("existing dictionary resume requires its trusted expected_sha256")
        return load_dictionary(
            directory, expected_sha256=expected_sha256, expected_contract=contract
        )
    if expected_sha256 is not None:
        raise ValueError("cannot resume a missing dictionary")
    if offset < N_CLUSTERS:
        raise ValueError(f"insufficient content rows: need at least {N_CLUSTERS}, got {offset}")
    x = np.concatenate(arrays).astype(np.float64, copy=False)
    with threadpool_limits(limits=1):
        scaler = StandardScaler(**SCALER_PARAMETERS)
        z = scaler.fit_transform(x)
        if not np.isfinite(z).all():
            raise ValueError("nonfinite standardized fit activations")
        estimator = KMeans(**KMEANS_PARAMETERS).fit(z)
    labels = estimator.labels_.astype("<i4", copy=False)
    if np.unique(labels).size != N_CLUSTERS:
        raise ValueError("fit did not populate all 50 fine clusters")
    saved = {
        "mean": scaler.mean_, "scale": scaler.scale_,
        "centers": estimator.cluster_centers_, "fit_labels": labels,
    }
    directory.mkdir(parents=True, exist_ok=True)
    bind_manifest(directory / "contract.json", contract)
    for name, array in saved.items():
        with (directory / f"{name}.npy").open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
    reloaded = _read_arrays(directory, offset)
    got = _predict(x, reloaded["mean"], reloaded["scale"], reloaded["centers"], chunk_size)
    if not np.array_equal(reloaded["fit_labels"], labels) or not np.array_equal(got, labels):
        raise ValueError("reloaded assignment differs from fit labels; dictionary NOT published")
    if digest(contract) != contract_hash or contract["implementation"] != _implementation():
        raise ValueError("fit contract or implementation changed during fitting")
    for record in records:
        if "activation_path" in record and file_hash(Path(record["activation_path"])) != (
            record["capture_manifest"]["activations_sha256"]
        ):
            raise ValueError("source activations changed during fitting; dictionary NOT published")
    manifest = {
        "format": FORMAT, "files": {name: file_hash(directory / name) for name in COMPONENTS},
        "certificate": {"fit_rows": offset, "exact_reloaded_labels": offset},
    }
    dictionary_hash = digest(manifest)
    bind_manifest(directory / "manifest.json", {**manifest, "dictionary_sha256": dictionary_hash})
    return load_dictionary(directory, expected_sha256=dictionary_hash)


def assign_response(
    dictionary: FrozenDictionary, capture: Capture, *, chunk_size: int = 1024
) -> dict:
    """Assign content rows from caller-attested CLEAN exact-ID replay, never injected states.

    An empty readout can use an evidenced (0, 3072) capture. Missing captures instead
    go to ``summarize_response(..., assignment=None, readout_error=...)``; no zeros
    are invented for failed replay. Shape alone never licenses this assignment.
    """
    _, array = _capture_record(
        capture, dictionary.contract["source"]["runtime_contract_sha256"]
    )
    if array is None:
        raise ValueError("assignment requires explicit exact-ID capture evidence and activations")
    content_labels = iter(dictionary.predict(array, chunk_size=chunk_size).tolist())
    return {
        "full_ids_sha256": digest(capture.generation["full_ids"]),
        "dictionary_sha256": dictionary.dictionary_sha256,
        "layer": LAYER, "n_clusters": N_CLUSTERS, "alignment": ALIGNMENT,
        "labels": [next(content_labels) if content else None
                   for content in capture.generation["content_mask"]],
    }
