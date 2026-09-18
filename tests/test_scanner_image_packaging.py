"""Standalone admission and normalization images must carry the default scanner."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGES = [ROOT / "services" / role / "Dockerfile" for role in ("ingestion", "worker")]


@pytest.mark.parametrize("dockerfile", IMAGES, ids=["ingestion", "worker"])
def test_default_scanner_is_built_and_installed_in_runtime_image(dockerfile):
    text = dockerfile.read_text()
    stages = re.split(r"(?m)^FROM ", text)[1:]
    builder = next(
        (stage for stage in stages if re.match(r"golang:[\d.]+-\w+ AS redactd-build\n", stage)),
        None,
    )
    assert builder is not None, "image lacks a pinned Go stage for the default redactd transport"
    assert "COPY tools/redactd/go.mod tools/redactd/go.sum ./" in builder
    assert "go mod download" in builder
    assert "COPY tools/redactd/ ./" in builder
    assert re.search(r"CGO_ENABLED=0 go build .*?-mod=readonly .*?-o /out/redactd \.", builder)
    runtime = stages[-1]
    assert "COPY --from=redactd-build /out/redactd /usr/local/bin/redactd" in runtime
    assert "COPY engine/ ./engine/" in runtime  # includes the shared rules file
    engine_version = re.search(
        r"github.com/zricethezav/gitleaks/v8 v([\d.]+)", (ROOT / "tools/redactd/go.mod").read_text()
    ).group(1)
    assert f"ARG GITLEAKS_VERSION={engine_version}" in runtime


def test_ingestion_and_worker_use_identical_scanner_builds():
    recipes = [p.read_text().split("FROM python:", 1)[0] for p in IMAGES]
    assert recipes[0] == recipes[1]
