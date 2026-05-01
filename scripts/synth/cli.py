"""CLI dispatch for the synth tool. Subcommands grow over plans 1-3."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.cache import DiskCache, default_cache_root
from scripts.synth.company_context import (
    CompanyContext,
    infer_company_context,
    load_company_context,
)
from scripts.synth.extractor.github_api import GithubClient
from scripts.synth.extractor.repo import RepoExtractor, RepoSignals
from scripts.synth.llm_client import LlmClient, LlmClientProtocol
from scripts.synth.profile import Profile, load_profile
from scripts.synth.world_model import merge_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser(
        "extract",
        help="Extract WorldModel from repos in a profile (no DB writes).",
    )
    extract.add_argument("--profile", required=True, type=str, help="Path to profile YAML.")
    extract.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write world_model.json (default: eval-datasets/<run-id>/).",
    )

    return parser


def _resolve_output_dir(profile: Profile, override: str | None) -> Path:
    if override:
        return Path(override)
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ") + f"-{profile.preset}-seed{profile.seed}"
    return Path("eval-datasets") / run_id


async def _extract_async(profile: Profile, out: Path) -> int:
    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    # Profile world_model knobs override defaults from spec §12.3
    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))

    from datetime import timedelta
    since = datetime.now(UTC).replace(microsecond=0) - timedelta(days=lookback_days)

    signals: list[RepoSignals] = []
    for repo in profile.repos:
        if repo.local_path is None:
            print(f"warn: repo {repo.url!r} has no local_path; skipping", file=sys.stderr)
            continue
        if gh_client is not None:
            sig = await extractor.extract(repo.url, repo.local_path, since=since, fetch_github=True)
        else:
            sig = extractor.extract_local(repo.url, repo.local_path, since=since)
        signals.append(sig)

    if gh_client is not None:
        await gh_client.close()

    if not signals:
        print("error: no repos extracted; check profile.repos[*].local_path", file=sys.stderr)
        return 3

    cc = await _resolve_company_context(profile, signals, out)

    wm = merge_world_model(
        signals=signals,
        company_name=cc.name,
        seed=profile.seed,
        min_threshold=min_threshold,
        max_personas=max_personas,
        now=datetime.now(UTC),
    )

    (out / "world_model.json").write_text(_dumps(wm))
    (out / "company_context.json").write_text(_dumps(cc))
    print(f"wrote {out}/world_model.json", file=sys.stderr)
    return 0


async def _resolve_company_context(
    profile: Profile,
    signals: list[RepoSignals],
    out: Path,
) -> CompanyContext:
    if profile.company_context_path is not None:
        return load_company_context(profile.company_context_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # No LLM available; fall back to a minimal stub. The user can
        # add company_context: ./<file> later for richer context.
        return CompanyContext(
            name=_infer_company_name_from_repos(signals),
            stage="unknown",
            headcount=0,
            inferred=True,
        )
    llm: LlmClientProtocol = LlmClient(api_key=api_key)
    try:
        readme_blob = "\n\n".join(
            r.content for sig in signals for r in sig.readmes if r.content
        )[:20_000]
        repo_descs = [s.description or s.url for s in signals]
        cc, raw_yaml = await infer_company_context(
            readme_blob=readme_blob,
            repo_descriptions=repo_descs,
            llm_client=llm,
            model="claude-opus-4-7",
        )
        (out / "inferred-company.yaml").write_text(raw_yaml)
        return cc
    finally:
        await llm.close()


def _infer_company_name_from_repos(signals: list[RepoSignals]) -> str:
    """Best-effort name when no LLM available: longest-common-prefix
    of repo URL owners; else 'unknown'.

    Bug-fix vs plan spec: skip empty owner segments that arise from
    non-github URL schemes like `repo://fake` where split("/") yields
    ["repo:", "", "fake"] and parts[-2] is "".
    """
    owners: set[str] = set()
    for sig in signals:
        # github.com/owner/repo  →  owner
        parts = sig.url.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2]:  # skip empty owner segments
            owners.add(parts[-2])
    if len(owners) == 1:
        return next(iter(owners))
    return "unknown"


def _dumps(obj) -> str:
    """Pretty JSON serializer that handles dataclasses + datetimes + Paths."""
    def encode(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: encode(getattr(v, k)) for k in (f.name for f in dataclasses.fields(v))}
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [encode(x) for x in v]
        if isinstance(v, dict):
            return {k: encode(val) for k, val in v.items()}
        return v
    return json.dumps(encode(obj), indent=2, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        return asyncio.run(_extract_async(profile, out))
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
