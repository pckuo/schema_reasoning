"""Step D analysis + figures: occupancy shift, specificity, dose-response, norm parity.

Reads ``inject/{generations,occupancy}.jsonl`` (+ optionally ``judged.jsonl`` and
``parity/parity.json``) and writes the figures and the summary tables that ``findings.md`` quotes.
Everything is **paired by scenario**: each cell ran the same 12 items in the same batch, and the
model is run-to-run deterministic (parity ``repeat`` check), so the per-item difference from the
alpha=0 baseline is the injection and nothing else.

Run::

    uv run python -m dev.schema_llm.explore.primitive_transplant.analyze --out <analysis dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

N_THEME = 35


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def paired(occ: list[dict], layer: int) -> dict:
    """``cell -> (theme_delta_matrix (n_items, 35), item order)`` against the alpha=0 baseline."""
    base = {r["scenario_id"]: np.array(r[f"occ_L{layer}"]) for r in occ if r["kind"] == "baseline"}
    out: dict[str, dict] = {}
    for r in occ:
        if r["kind"] == "baseline":
            continue
        e = out.setdefault(
            r["cell"],
            {
                "delta": [],
                "items": [],
                "theme": r["theme"],
                "kind": r["kind"],
                "layer": r["layer"],
                "alpha_rel": r["alpha_rel"],
                "occ": [],
            },
        )
        e["delta"].append(np.array(r[f"occ_L{layer}"]) - base[r["scenario_id"]])
        e["occ"].append(np.array(r[f"occ_L{layer}"]))
        e["items"].append(r["scenario_id"])
    for e in out.values():
        e["delta"] = np.stack(e["delta"])
        e["occ"] = np.stack(e["occ"])
    return out


def _boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(occ: list[dict], layer: int) -> dict:
    """Per cell: on-target shift, the largest off-target shift, and both against the random-vector
    null built from the control cells' per-item shifts."""
    cells = paired(occ, layer)
    base = np.stack([np.array(r[f"occ_L{layer}"]) for r in occ if r["kind"] == "baseline"]).mean(0)
    ctrl = (
        np.concatenate([e["delta"] for e in cells.values() if e["kind"] == "random"])
        if any(e["kind"] == "random" for e in cells.values())
        else None
    )
    rows = []
    for name, e in cells.items():
        d = e["delta"]
        mean = d.mean(0)
        row = {
            "cell": name,
            "kind": e["kind"],
            "layer": e["layer"],
            "theme": e["theme"],
            "alpha_rel": e["alpha_rel"],
            "n_items": int(d.shape[0]),
            "total_abs_shift": round(float(np.abs(mean).sum()), 5),
        }
        if e["theme"] is not None:
            t = e["theme"]
            on = d[:, t]
            lo, hi = _boot_ci(on)
            off = np.abs(np.delete(mean, t))
            row |= {
                "on_target_delta": round(float(on.mean()), 5),
                "on_target_ci": [round(lo, 5), round(hi, 5)],
                "on_target_frac_items_up": round(float((on > 0).mean()), 3),
                "max_off_target_delta": round(float(off.max()), 5),
                "off_target_theme": int(np.argsort(-np.abs(mean))[0]),
                "specificity_rank": int((np.abs(mean) > abs(mean[t])).sum()) + 1,
                "baseline_occupancy": round(float(base[t]), 5),
                "injected_occupancy": round(float(e["occ"].mean(0)[t]), 5),
            }
            if ctrl is not None:
                cm, cs = ctrl[:, t].mean(), ctrl[:, t].std() + 1e-9
                row["z_vs_random_control"] = round(float((on.mean() - cm) / cs), 3)
        rows.append(row)
    out = {"layer": layer, "cells": sorted(rows, key=lambda r: r["cell"])}
    out["column_contrast"] = _column_contrast(cells)
    return out


def _column_contrast(cells: dict) -> list[dict]:
    """The specificity statistic that controls for 'theme T is simply easy to move': within the
    measured column for T, the shift when T was injected minus the mean shift when some OTHER
    theme was injected at the same layer and dose. Positive = the direction is doing something
    T-specific; ~0 = T moves whatever you inject."""
    peers: dict[tuple[int, float], list] = defaultdict(list)
    for e in cells.values():
        if e["kind"] == "theme":
            peers[(e["layer"], e["alpha_rel"])].append(e)
    out = []
    for (lay, alpha), group in sorted(peers.items()):
        if len(group) < 2:
            continue
        for e in group:
            t = e["theme"]
            others = [g["delta"].mean(0)[t] for g in group if g["theme"] != t]
            ctrls = [
                g["delta"].mean(0)[t]
                for g in cells.values()
                if g["kind"] == "random" and g["layer"] == lay and g["alpha_rel"] == alpha
            ]
            out.append(
                {
                    "layer": lay,
                    "alpha_rel": alpha,
                    "theme": t,
                    "on_diagonal": round(float(e["delta"].mean(0)[t]), 5),
                    "mean_off_diagonal_same_column": round(float(np.mean(others)), 5),
                    "diagonal_minus_column": round(
                        float(e["delta"].mean(0)[t] - np.mean(others)), 5
                    ),
                    "random_same_column": (round(float(np.mean(ctrls)), 5) if ctrls else None),
                    "n_peers": len(others),
                }
            )
    return out


def fig_occupancy_shift(
    occ: list[dict], layer: int, spec_layer: int, alpha: float, out: Path
) -> None:
    """Per-theme occupancy shift for each single-primitive injection at the specificity layer."""
    cells = paired(occ, layer)
    sel = [
        e
        for e in cells.values()
        if e["layer"] == spec_layer and e["alpha_rel"] == alpha and e["kind"] == "theme"
    ]
    ctrl = [
        e
        for e in cells.values()
        if e["kind"] == "random" and e["layer"] == spec_layer and e["alpha_rel"] == alpha
    ]
    if not sel:
        return
    themes = sorted(
        {t for e in sel for t in np.nonzero(e["delta"].mean(0))[0]} | {e["theme"] for e in sel}
    )
    names = C.theme_names()
    series = sorted(sel, key=lambda v: v["theme"]) + ctrl
    fig, ax = plt.subplots(figsize=(1.3 * len(themes) + 4, 0.45 * len(series) + 3.4))
    w = 0.86 / len(series)
    n_ctrl = 0
    for i, e in enumerate(series):
        m = e["delta"].mean(0)
        se = e["delta"].std(0) / np.sqrt(e["delta"].shape[0])
        x = np.arange(len(themes)) + i * w - 0.43
        if e["kind"] == "theme":
            lab = f"inject t{e['theme']}"
        else:
            lab = f"random unit vector {n_ctrl}"
            n_ctrl += 1
        ax.bar(
            x,
            [m[t] for t in themes],
            w,
            yerr=[se[t] for t in themes],
            label=lab,
            color=None if e["kind"] == "theme" else "0.4",
            hatch=None if e["kind"] == "theme" else "//",
            error_kw={"lw": 0.6},
        )
        if e["kind"] == "theme":
            ax.plot(x[themes.index(e["theme"])], m[e["theme"]], "k*", ms=9)
    ax.set_xticks(range(len(themes)))
    ax.set_xticklabels(
        [f"t{t}\n{names[t][:18]}" for t in themes], fontsize=7, rotation=30, ha="right"
    )
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel(f"$\\Delta$ occupancy vs $\\alpha$=0 (read at L{layer})")
    ax.set_title(
        f"Occupancy shift per theme -- inject at L{spec_layer}, "
        f"$\\alpha_{{rel}}$={alpha} (star = injected theme)"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_specificity(occ: list[dict], layer: int, spec_layer: int, alpha: float, out: Path) -> None:
    """Injected x measured theme matrix. A causal, specific primitive puts its mass on the
    diagonal; a general perturbation lights up whole columns."""
    cells = paired(occ, layer)
    sel = sorted(
        [
            e
            for e in cells.values()
            if e["layer"] == spec_layer and e["alpha_rel"] == alpha and e["kind"] == "theme"
        ],
        key=lambda e: e["theme"],
    )
    ctrl = [
        e
        for e in cells.values()
        if e["kind"] == "random" and e["layer"] == spec_layer and e["alpha_rel"] == alpha
    ]
    if not sel:
        return
    inj = [e["theme"] for e in sel]
    rows_e = sel + ctrl
    meas = sorted(
        {t for e in sel for t in np.nonzero(np.abs(e["delta"].mean(0)) > 1e-9)[0]} | set(inj)
    )
    m = np.array([[e["delta"].mean(0)[t] for t in meas] for e in rows_e])
    names = C.theme_names()
    v = np.abs(m).max()
    ylab = [f"t{t} {names[t][:22]}" for t in inj] + [f"random {i}" for i in range(len(ctrl))]
    fig, ax = plt.subplots(figsize=(0.85 * len(meas) + 5, 0.55 * len(rows_e) + 3.4))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(meas)))
    ax.set_xticklabels([f"t{t}" for t in meas], fontsize=8)
    ax.set_yticks(range(len(rows_e)))
    ax.set_yticklabels(ylab, fontsize=8)
    if ctrl:
        ax.axhline(len(inj) - 0.5, color="k", lw=1.5)
    for i in range(len(rows_e)):
        for j in range(len(meas)):
            ax.text(
                j,
                i,
                f"{m[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="k" if abs(m[i, j]) < 0.6 * v else "w",
            )
        if i < len(inj) and inj[i] in meas:
            j = meas.index(inj[i])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=2))
    ax.set_xlabel("measured theme (clean re-forward)")
    ax.set_ylabel("injected theme")
    ax.set_title(
        f"Injected x measured $\\Delta$ occupancy -- inject L{spec_layer}, "
        f"$\\alpha_{{rel}}$={alpha}, read L{layer}\n"
        f"(boxed = on-diagonal; rows below the rule are matched-norm random directions)"
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_dose(occ: list[dict], gens: list[dict], layer: int, out: Path) -> None:
    """Dose-response for the focus themes: on-target shift, total absolute shift, and length."""
    cells = paired(occ, layer)
    by_theme: dict[int, list] = defaultdict(list)
    for e in cells.values():
        if e["kind"] == "theme":
            by_theme[e["theme"]].append(e)
    focus = [t for t, es in by_theme.items() if len({e["alpha_rel"] for e in es}) > 1]
    if not focus:
        return
    lens = defaultdict(list)
    for g in gens:
        lens[g["cell"]].append(len(g["text"]))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    names = C.theme_names()
    for t in sorted(focus):
        es = sorted(
            [e for e in by_theme[t] if e["layer"] == _spec_layer_of(by_theme[t])],
            key=lambda e: e["alpha_rel"],
        )
        a = [e["alpha_rel"] for e in es]
        on = [e["delta"].mean(0)[t] for e in es]
        se = [e["delta"][:, t].std() / np.sqrt(e["delta"].shape[0]) for e in es]
        ax[0].errorbar([0] + a, [0] + on, yerr=[0] + se, marker="o", label=f"t{t} {names[t][:20]}")
        ax[1].plot([0] + a, [0] + [float(np.abs(e["delta"].mean(0)).sum()) for e in es], "o-")
        cellnames = [f"L{e['layer']}_t{t}_a{e['alpha_rel']}" for e in es]
        base_len = float(np.mean(lens["baseline"])) if lens["baseline"] else np.nan
        ax[2].plot([0] + a, [base_len] + [float(np.mean(lens[c])) for c in cellnames], "o-")
    for e in cells.values():
        if e["kind"] == "random":
            ax[0].axhline(e["delta"].mean(0).max(), ls=":", c="gray", lw=0.8)
    ax[0].set_ylabel("$\\Delta$ on-target occupancy")
    ax[1].set_ylabel("total |$\\Delta$| occupancy (all themes)")
    ax[2].set_ylabel("mean response length (chars)")
    for a_ in ax:
        a_.set_xlabel("$\\alpha_{rel}$ (fraction of median $\\|h\\|_L$)")
        a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Dose-response at the specificity layer (read at L{layer})")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _spec_layer_of(es: list[dict]) -> int:
    counts: dict[int, int] = defaultdict(int)
    for e in es:
        counts[e["layer"]] += 1
    return max(counts, key=lambda k: counts[k])


def fig_norms(parity: dict, out: Path) -> None:
    """Runtime vs stored ``||h||`` per layer -- the dose denominator, and the calibration that says
    the injection runs in the same space the dictionary was built in."""
    norm = parity.get("norm") or {}
    layers = sorted(int(k) for k in norm)
    med = [norm[str(lay)]["median"] for lay in layers]
    p10 = [norm[str(lay)]["p10"] for lay in layers]
    p90 = [norm[str(lay)]["p90"] for lay in layers]
    run_by: dict[int, list] = defaultdict(list)
    dev_by: dict[int, list] = defaultdict(list)
    for r in parity.get("residual", []):
        if "median_norm_runtime" in r:
            run_by[r["layer"]].append(r["median_norm_runtime"])
            dev_by[r["layer"]].append(r["max_abs"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].fill_between(layers, p10, p90, alpha=0.2, label="stored p10-p90 (216 items)")
    ax[0].plot(layers, med, "o-", label="stored median")
    if run_by:
        rl = sorted(run_by)
        ax[0].plot(
            rl,
            [np.median(run_by[lay]) for lay in rl],
            "kx",
            ms=9,
            label="runtime re-forward (median)",
        )
    ax[0].set_yscale("log")
    ax[0].set_ylabel("$\\|h\\|$ (response tokens)")
    ax[0].set_title(
        f"residual norm grows {med[-1] / med[0]:.1f}x from L{layers[0]} to L{layers[-1]}"
    )
    ax[0].legend(fontsize=7)
    if dev_by:
        dl = sorted(dev_by)
        ax[1].plot(dl, [max(dev_by[lay]) for lay in dl], "s-", c="tab:red")
    ax[1].set_ylabel("max |runtime - stored| (fp16 units)")
    ax[1].set_title("re-forward parity per layer")
    ax[1].set_ylim(-0.01, max(0.02, max((max(v) for v in dev_by.values()), default=0.02)))
    for a_ in ax:
        a_.set_xlabel("layer")
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def degeneration(gens: list[dict], occ: list[dict]) -> list[dict]:
    """Length + 4-gram repetition + off-manifold fraction per cell (the fluency guard)."""
    off = defaultdict(list)
    ntok = defaultdict(list)
    for r in occ:
        for k in r:
            if k.startswith("offman_L") and r[k] is not None:  # None = empty response
                off[(r["cell"], k)].append(r[k])
        ntok[r["cell"]].append(r["n_response_tokens"])
    rows = []
    by_cell = defaultdict(list)
    for g in gens:
        by_cell[g["cell"]].append(g)
    for cell, gs in by_cell.items():
        rep = [_repeat_frac(g["text"]) for g in gs]
        row = {
            "cell": cell,
            "kind": gs[0]["kind"],
            "layer": gs[0]["layer"],
            "theme": gs[0]["theme"],
            "alpha_rel": gs[0]["alpha_rel"],
            "mean_chars": round(float(np.mean([len(g["text"]) for g in gs])), 1),
            "mean_tokens": round(float(np.mean(ntok[cell])), 1) if ntok[cell] else None,
            "mean_repeat_frac": round(float(np.mean(rep)), 4),
            "max_repeat_frac": round(float(np.max(rep)), 4),
        }
        row["n_empty_responses"] = int(sum(n < 1 for n in ntok[cell]))
        for key, vals in off.items():
            if key[0] == cell and vals:
                row[key[1]] = round(float(np.mean(vals)), 5)
        rows.append(row)
    return sorted(rows, key=lambda r: r["cell"])


def _repeat_frac(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) <= n:
        return 0.0
    grams = [" ".join(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def run(out_dir: Path, read_layer: int, spec_layer: int, alpha: float) -> dict:
    inj = out_dir / "inject"
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    occ = _read(inj / "occupancy.jsonl")
    gens = _read(inj / "generations.jsonl")
    summary = {
        "read_layer": read_layer,
        "spec_layer": spec_layer,
        "alpha_rel": alpha,
        "occupancy": summarise(occ, read_layer),
        "degeneration": degeneration(gens, occ),
    }
    for lay in sorted({int(k[len("occ_L") :]) for k in occ[0] if k.startswith("occ_L")}):
        summary[f"occupancy_L{lay}"] = summarise(occ, lay)
    fig_occupancy_shift(occ, read_layer, spec_layer, alpha, figs / "occupancy_shift.png")
    fig_specificity(occ, read_layer, spec_layer, alpha, figs / "specificity_matrix.png")
    fig_dose(occ, gens, read_layer, figs / "dose_response.png")
    parity_p = out_dir / "parity" / "parity.json"
    if parity_p.exists():
        fig_norms(json.loads(parity_p.read_text(encoding="utf-8")), figs / "norm_parity.png")
    judged = inj / "judged.jsonl"
    if judged.exists():
        summary["outcomes"] = outcome_table(_read(judged))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"-> {out_dir}/summary.json + {figs}")
    return summary


def outcome_table(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    out = []
    for cell, rs in by.items():
        viol = [r for r in rs if r.get("has_violation")]
        out.append(
            {
                "cell": cell,
                "n": len(rs),
                "parse_ok": round(float(np.mean([bool(r["parse_ok"]) for r in rs])), 3),
                "detected": round(float(np.mean([bool(r["detected"]) for r in rs])), 3),
                "n_violation": len(viol),
                "localized": (
                    round(float(np.mean([bool(r["localized"]) for r in viol])), 3) if viol else None
                ),
            }
        )
    return sorted(out, key=lambda r: r["cell"])


def main() -> None:
    p = argparse.ArgumentParser(description="step D analysis + figures")
    p.add_argument("--out", required=True, help="the analysis dir holding inject/ and parity/")
    p.add_argument("--read-layer", type=int, default=24)
    p.add_argument("--spec-layer", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.2)
    a = p.parse_args()
    run(Path(a.out), a.read_layer, a.spec_layer, a.alpha)


if __name__ == "__main__":
    main()
