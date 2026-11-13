"""Offline checks for the standalone D1/D2 generator and its continued name pool."""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from dev.schema_llm.binding.dynamics.design import build_wave3, build_wave4
from dev.schema_llm.binding.dynamics.prose import (
    build_continue_prompt,
    build_d2_branch_prompt,
    build_d2_prefix_prompt,
    build_seed_prompt,
)
from dev.schema_llm.binding.dynamics.run import pinned_depth, seed_arm
from dev.schema_llm.core.spec import load_schema


@pytest.mark.parametrize("design", ["D1", "D2"])
def test_two_wave_generation_contract(design, tmp_path):
    first = build_wave3(design, seed=20260914)
    first_items = [item for family in first for item in family.items]
    # Resume reads accepted records' casts; passage generation is deliberately unnecessary here.
    (tmp_path / "items.jsonl").write_text(
        "".join(json.dumps(item.to_dict()) + "\n" for item in first_items), encoding="utf-8"
    )
    second = build_wave4(design, tmp_path, seed=20260915)
    second_items = [item for family in second for item in family.items]
    assert len(first_items) == len(second_items) == 150
    assert not ({f.family_id for f in first} & {f.family_id for f in second})
    assert Counter(i.controls["n_segments"] for i in second_items) == {8: 78, 6: 72}
    assert {p.name for i in second_items for p in i.cast} <= {
        p.name for i in first_items for p in i.cast
    }
    for family in first + second:
        schema = load_schema(family.schema)
        if design == "D2":
            prompts = [build_d2_prefix_prompt(schema, family.items[0])]
            for item in family.items:
                prefix = ["A neutral segment."] * max(item.controls["ambiguous_segments"])
                prompts.append(build_d2_branch_prompt(schema, item, prefix))
        else:
            seed = next(i for i in family.items if i.controls["arm"] == seed_arm(family))
            prompts = [build_seed_prompt(schema, seed)]
            for item in family.items:
                depth = pinned_depth(item, seed)
                if depth:
                    prefix = ["A neutral segment."] * depth
                    prompts.append(build_continue_prompt(schema, item, prefix))
        assert all(prompt and "<<" not in prompt for prompt in prompts)


@pytest.mark.parametrize("design", ["D1", "D2"])
@pytest.mark.parametrize("wave", [3, 4])
def test_original_stage_script_generation_only(design, wave, tmp_path):
    root = Path(__file__).resolve().parents[4]
    script_name = f"run_binding_dynamics_{design.lower()}.sh"
    scripts = tmp_path / "scripts/schema_llm"
    scripts.mkdir(parents=True)
    script = scripts / script_name
    shutil.copy2(root / "scripts/schema_llm" / script_name, script)
    (tmp_path / "code").mkdir()
    calls = tmp_path / "calls.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['GENERATION_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    name = f"binding_dyn_{design.lower()}_v3" + ("w2" if wave == 4 else "")
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    env.update(
        GENERATION_TEST_CALLS=str(calls), STEP="gen", WAVE=str(wave), NAME=name,
        DESIGN=design, POOL="0", MODEL="test-deployment", REPAIR="0",
    )
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    # Only the generation command may run: collection/capture/export scripts are absent here.
    run, = [json.loads(line) for line in calls.read_text().splitlines()]
    assert run[:7] == ["run", "--extra", "api", "--directory", "code", "python", "-m"]
    assert run[7] == "dev.schema_llm.binding.dynamics.run"
    assert run[run.index("--design") + 1] == design
    assert run[run.index("--wave") + 1] == str(wave)
    assert run[run.index("--out") + 1] == f"../results/schema_llm/data/{name}"
    assert run[run.index("--model") + 1] == "test-deployment"
