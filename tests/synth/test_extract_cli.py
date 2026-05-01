"""End-to-end: run `python -m scripts.synth extract --profile <yaml>`
on a profile pointing at the tmp_repo fixture; assert world_model.json
is written and contains the expected high-level shape."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_extract_writes_world_model_json(tmp_repo: Path, tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        f"""
customer_id: cust-eval-fake-01
repos:
  - url: repo://fake
    local_path: {tmp_repo}
preset: tiny-test
seed: 7
world_model:
  min_commits_per_persona: 1
  topic_pool_lookback_days: 9999
""".strip()
    )

    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = ""  # extract doesn't need a real key (no LLM unless company_context auto-infers)

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.synth", "extract",
            "--profile", str(profile),
            "--output-dir", str(out_dir),
        ],
        check=False, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    world_model_path = out_dir / "world_model.json"
    assert world_model_path.exists()
    wm = json.loads(world_model_path.read_text())

    assert wm["company_name"]
    assert wm["seed"] == 7
    assert {p["display_name"] for p in wm["people"]} >= {"Alice", "Bob", "Carol"}
    assert {s["name"] for s in wm["services"]} >= {"payments", "billing", "fake-repo"}
