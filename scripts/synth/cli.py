"""CLI dispatch for the synth tool.

Plan 1: extract subcommand (WorldModel dump only, no DB).
Plan 2: init / run / clean subcommands.

Plan 2 commands default to local-files mode. The --integrate flag opts into
DB + R2 writes (requires a prior `synth init`).

Plan 3 additions: LlmClientConfig, LlmClients, CachingLlmClient, build_llm_clients.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from scripts.synth.archetypes.base import ScenarioSpec
from scripts.synth.bootstrap import clean_tenant, init_tenant
from scripts.synth.cache import DiskCache, default_cache_root
from scripts.synth.company_context import (
    CompanyContext,
    infer_company_context,
    load_company_context,
)
from scripts.synth.extractor.github_api import GithubClient
from scripts.synth.extractor.repo import RepoExtractor, RepoSignals
from scripts.synth.llm.anthropic_client import AnthropicClient
from scripts.synth.llm.base import (
    LlmClientProtocol,
    LlmRequest,
    LlmResponse,
    Provider,
    provider_from_model,
)
from scripts.synth.llm.cache import PromptCache
from scripts.synth.llm.fixtures import FixtureStore
from scripts.synth.llm.gemini_client import GeminiClient
from scripts.synth.llm.mock_client import MockLlmClient
from scripts.synth.llm.planner import LLMPlanner
from scripts.synth.llm.writer import LLMWriter
from scripts.synth.output.eval_artifacts import (
    write_docs_index,
    write_manifest,
    write_profile,
    write_questions_jsonl,
    write_scenarios,
    write_warnings,
)
from scripts.synth.output.writer import IngestionWriter
from scripts.synth.ownership import build_ownership_index
from scripts.synth.profile import Profile, load_profile
from scripts.synth.scenarios import TimeWindow, run_scenarios
from scripts.synth.validator import validate_name_only
from scripts.synth.world_model import merge_world_model

# Default fixture root for MockLlmClient replay mode — relative to repo root.
_DEFAULT_FIXTURE_ROOT = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "synth_llm"

# Canonical model roster for the three LLM roles.  Profile `llm:` section can
# override individual keys; these are the fallback defaults.
_LLM_DEFAULTS: dict[str, str] = {
    "planner_model": "claude-opus-4-7",
    "writer_model": "claude-sonnet-4-6",
    "validator_model": "claude-haiku-4-5-20251001",
}


# ---------------------------------------------------------------------------
# Plan 3 — LLM client dataclasses + CachingLlmClient + build_llm_clients
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmClientConfig:
    """Configuration bundle passed to build_llm_clients."""

    llm_cfg: dict
    mock_llm: bool
    no_llm_cache: bool
    record_llm: bool
    fixture_root: Path | None = None
    cache_root: Path | None = None


@dataclass
class LlmClients:
    """Container for the three role clients + their resolved providers."""

    planner_client: LlmClientProtocol
    writer_client: LlmClientProtocol
    validator_client: LlmClientProtocol
    planner_provider: Provider
    writer_provider: Provider
    validator_provider: Provider


class CachingLlmClient:
    """LlmClientProtocol wrapper that consults PromptCache before calling the inner client."""

    def __init__(self, inner: LlmClientProtocol, cache: PromptCache, provider: Provider) -> None:
        self._inner = inner
        self._cache = cache
        self._provider = provider

    async def generate(self, req: LlmRequest) -> LlmResponse:
        cached = await self._cache.get(self._provider, req, schema_json=None)
        if cached is not None:
            return LlmResponse(text=cached.get("text", ""))
        resp = await self._inner.generate(req)
        await self._cache.put(self._provider, req, schema_json=None, response_dict={"text": resp.text})
        return resp

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        cached = await self._cache.get(self._provider, req, schema_json)
        if cached is not None:
            return cached
        result = await self._inner.generate_structured(req, schema)
        await self._cache.put(self._provider, req, schema_json, result)
        return result

    async def close(self) -> None:
        await self._inner.close()


def build_llm_clients(cfg: LlmClientConfig) -> LlmClients:
    """Construct the three role clients according to the flag combination in cfg.

    Routing logic:
      - mock_llm=True  → all three clients are MockLlmClient(replay). No API keys needed.
      - record_llm=True → wrap each real client with MockLlmClient(record). Raises if keys absent.
      - no_llm_cache=True → use raw real clients (no CachingLlmClient wrapper).
      - default         → wrap each real client in CachingLlmClient.
    """
    planner_model = cfg.llm_cfg["planner_model"]
    writer_model = cfg.llm_cfg["writer_model"]
    validator_model = cfg.llm_cfg["validator_model"]

    planner_provider = provider_from_model(planner_model)
    writer_provider = provider_from_model(writer_model)
    validator_provider = provider_from_model(validator_model)

    # --- mock path: no API keys, pure fixture replay ---
    if cfg.mock_llm:
        store = FixtureStore(cfg.fixture_root or _DEFAULT_FIXTURE_ROOT)
        mock = MockLlmClient(store=store, mode="replay")
        return LlmClients(
            planner_client=mock,
            writer_client=mock,
            validator_client=mock,
            planner_provider=planner_provider,
            writer_provider=writer_provider,
            validator_provider=validator_provider,
        )

    # --- record path: validate required API keys up front ---
    if cfg.record_llm:
        needs_anthropic = any(
            p == Provider.ANTHROPIC
            for p in (planner_provider, writer_provider, validator_provider)
        )
        needs_google = any(
            p == Provider.GEMINI
            for p in (planner_provider, writer_provider, validator_provider)
        )
        if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY required for --record-llm with Anthropic models"
            )
        if needs_google and not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY required for --record-llm with Gemini models"
            )

    # --- build one real client per role ---
    fixture_root = cfg.fixture_root or _DEFAULT_FIXTURE_ROOT

    def _make_client(model: str, provider: Provider) -> LlmClientProtocol:
        if provider == Provider.ANTHROPIC:
            inner: LlmClientProtocol = AnthropicClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        else:
            inner = GeminiClient(api_key=os.environ.get("GOOGLE_API_KEY", ""))

        if cfg.record_llm:
            store = FixtureStore(fixture_root)
            return MockLlmClient(store=store, mode="record", real_client=inner)

        if cfg.no_llm_cache:
            return inner

        # default: wrap in caching layer
        cache = PromptCache(cfg.cache_root or default_cache_root("llm"))
        return CachingLlmClient(inner=inner, cache=cache, provider=provider)

    return LlmClients(
        planner_client=_make_client(planner_model, planner_provider),
        writer_client=_make_client(writer_model, writer_provider),
        validator_client=_make_client(validator_model, validator_provider),
        planner_provider=planner_provider,
        writer_provider=writer_provider,
        validator_provider=validator_provider,
    )


def _build_args(argv: list[str]) -> argparse.Namespace:
    """Thin wrapper: parse *argv* with build_parser() and return the Namespace.

    Intended for in-process test invocations so tests don't shell out.
    """
    return build_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract — Plan 1, unchanged
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

    # init — Plan 2
    init = sub.add_parser(
        "init",
        help="Bootstrap a synthetic tenant: customers row + bucket + integration_tokens stubs.",
    )
    init.add_argument("--profile", required=True, type=str)

    # run — Plan 2
    run = sub.add_parser(
        "run",
        help="Run scenarios for a profile. Local files by default; --integrate writes to DB+R2.",
    )
    run.add_argument("--profile", required=True, type=str)
    run.add_argument("--output-dir", type=str, default=None)
    run.add_argument(
        "--integrate",
        action="store_true",
        help="Also write to R2 + ingestion_queue. Requires prior `synth init`.",
    )
    run.add_argument(
        "--reset",
        action="store_true",
        help="Call `synth clean` before running.",
    )
    run.add_argument(
        "--time-window",
        type=str,
        default=None,
        help="Override profile time_window.days (e.g., 30d, 14d). Default: 30d.",
    )
    run.add_argument(
        "--archetypes",
        type=str,
        default=None,
        help="Comma-separated archetype names to restrict the run.",
    )
    run.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Per-archetype scenario cap (debug).",
    )
    run.add_argument("--verbose", action="store_true")
    # Plan 3 LLM flags (run subcommand only)
    run.add_argument(
        "--mock-llm",
        action="store_true",
        default=False,
        help="Use MockLlmClient (fixture replay). No API keys required.",
    )
    run.add_argument(
        "--no-llm-cache",
        action="store_true",
        default=False,
        help="Bypass PromptCache; always call the real LLM API.",
    )
    run.add_argument(
        "--record-llm",
        action="store_true",
        default=False,
        help="Record real LLM responses to fixtures for later replay.",
    )

    # clean — Plan 2
    clean = sub.add_parser(
        "clean",
        help="Tear down a synthetic tenant. Refuses non-synth customer prefixes.",
    )
    clean.add_argument("--customer", required=True, type=str)

    # allow-seed — Plan 4
    allow_seed = sub.add_parser(
        "allow-seed",
        help="Toggle customers.metadata.allow_synth_seed=true for a real-shape tenant.",
    )
    allow_seed.add_argument("--customer", required=True, type=str)

    # seed — Plan 4
    seed = sub.add_parser(
        "seed",
        help="Replay canonical synthetic envelopes against an existing customer.",
    )
    seed.add_argument("--customer", required=True, type=str)
    seed.add_argument(
        "--allow-non-sandbox",
        action="store_true",
        default=False,
        help="Escape hatch: seed a non-eval-prefix tenant without setting "
             "metadata.allow_synth_seed first. Prompts for typed confirmation.",
    )
    seed.add_argument(
        "--canonical-dir",
        type=str,
        default="scripts/synth/canonical/v1",
        help="Path to the canonical corpus (default: scripts/synth/canonical/v1).",
    )

    return parser


def _resolve_output_dir(profile: Profile, override: str | None) -> Path:
    if override:
        return Path(override)
    run_id = (
        datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        + f"-{profile.preset}-seed{profile.seed}"
    )
    return Path("eval-datasets") / run_id


def _parse_time_window(arg: str | None, profile: Profile) -> TimeWindow:
    """CLI --time-window beats profile.time_window.days; both default to 30."""
    if arg is not None:
        days = int(arg.rstrip("d"))
    else:
        cfg = profile.raw.get("time_window") or {}
        days = int(cfg.get("days", 30))
    end = datetime.now(UTC).replace(microsecond=0)
    return TimeWindow(end=end, days=days)


# ---------------------------------------------------------------------------
# extract — Plan 1 (unchanged)
# ---------------------------------------------------------------------------


async def _extract_async(profile: Profile, out: Path) -> int:
    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))
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
        return CompanyContext(
            name=_infer_company_name_from_repos(signals),
            stage="unknown",
            headcount=0,
            inferred=True,
        )
    llm: LlmClientProtocol = AnthropicClient(api_key=api_key)
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


# ---------------------------------------------------------------------------
# Plan 2 helpers — DB / bucket connection
# ---------------------------------------------------------------------------


async def _open_db_and_bucket():
    """Construct (db_pool, bucket) for integrate mode. Pulls from settings.

    Adaptation from plan spec: plan uses `await get_pool()` but get_pool()
    is synchronous and raises DatabaseUnavailable if not initialized. We use
    `await init_pool()` which initializes the pool from settings and returns it.
    """
    from shared.db import init_pool  # type: ignore[import-untyped]
    from shared.storage import ObjectStore  # type: ignore[import-untyped]

    db = await init_pool()
    bucket = ObjectStore()
    return db, bucket


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


async def _init_async(profile: Profile) -> int:
    db, bucket = await _open_db_and_bucket()
    try:
        await init_tenant(profile, db, bucket)
        print(f"initialized tenant {profile.customer_id}", file=sys.stderr)
        return 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


async def _clean_async(customer_id: str) -> int:
    if not customer_id.startswith(("cust-eval-", "cust-synth-")):
        print(
            f"error: refuse to clean non-synthetic customer: {customer_id!r}",
            file=sys.stderr,
        )
        return 4
    db, bucket = await _open_db_and_bucket()
    try:
        await clean_tenant(customer_id, db, bucket)
        print(f"cleaned tenant {customer_id}", file=sys.stderr)
        return 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# allow-seed — Plan 4
# ---------------------------------------------------------------------------


async def _allow_seed_async(args) -> int:
    """CLI handler for `synth allow-seed`. Returns process exit code."""
    from scripts.synth.seed import set_allow_synth_seed

    db, _bucket = await _open_db_and_bucket()
    try:
        try:
            await set_allow_synth_seed(args.customer, db)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        await db.close()

    print(f"metadata.allow_synth_seed=true for {args.customer}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# seed — Plan 4
# ---------------------------------------------------------------------------


async def _seed_async(args) -> int:
    """CLI handler for `synth seed`. Returns process exit code.

    Gate stack order (cheap → expensive):
      1. Customer exists in DB
      2. Path 1 (metadata flag) OR Path 2 (--allow-non-sandbox + typed confirm)
      3. Canonical fixtures present
      4. Execute seed
    """
    import json as _json
    from pathlib import Path
    from scripts.synth.seed import (
        is_seed_eligible,
        prompt_typed_confirm,
        seed_tenant,
        MissingCanonicalError,
    )

    db, bucket = await _open_db_and_bucket()
    try:
        # Gate 1: customer must exist; fetch metadata while we're at it.
        row = await db.fetchrow(
            "SELECT metadata FROM customers WHERE customer_id = $1",
            args.customer,
        )
        if row is None:
            print(
                f"error: customer {args.customer!r} not found in customers table; "
                f"create the tenant via prbe-backend signup first",
                file=sys.stderr,
            )
            return 2

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = _json.loads(metadata)

        # Gate 2: Path 1 (flag) OR Path 2 (escape hatch).
        if not is_seed_eligible(args.customer, metadata):
            if not args.allow_non_sandbox:
                print(
                    f"error: customer {args.customer!r} is not seed-eligible. "
                    f"Either run 'synth allow-seed --customer {args.customer}' "
                    f"first, or pass --allow-non-sandbox to seed one-off.",
                    file=sys.stderr,
                )
                return 2
            # Path 2: escape hatch requires typed confirm.
            if not prompt_typed_confirm(args.customer):
                print(
                    f"error: confirmation mismatch; expected "
                    f"{args.customer!r}. No data written.",
                    file=sys.stderr,
                )
                return 2

        # Gate 3 + execute (MissingCanonicalError raised by seed_tenant).
        try:
            result = await seed_tenant(
                customer_id=args.customer,
                canonical_dir=Path(args.canonical_dir),
                db=db,
                bucket=bucket,
            )
        except MissingCanonicalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"seeded {result.envelopes_processed} envelopes "
            f"({result.r2_uploaded} uploaded, {result.queued} newly queued) "
            f"for {args.customer}",
            file=sys.stderr,
        )
        return 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _run_async(profile: Profile, out: Path, args) -> int:
    started_at = datetime.now(UTC)

    if args.reset and args.integrate:
        rc = await _clean_async(profile.customer_id)
        if rc != 0:
            return rc  # propagate clean failure
    elif args.reset:
        print("warn: --reset has no effect in local mode (no DB to clean); ignoring", file=sys.stderr)

    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))
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
    world = merge_world_model(
        signals=signals,
        company_name=cc.name,
        seed=profile.seed,
        min_threshold=min_threshold,
        max_personas=max_personas,
        now=datetime.now(UTC),
    )
    ownership = build_ownership_index(signals, world)
    time_window = _parse_time_window(args.time_window, profile)

    # Plan 3 — build LLM clients and the Planner/Writer/Validator helpers
    llm_cfg: dict = {**_LLM_DEFAULTS, **(profile.raw.get("llm") or {})}
    client_cfg = LlmClientConfig(
        llm_cfg=llm_cfg,
        mock_llm=args.mock_llm,
        no_llm_cache=args.no_llm_cache,
        record_llm=args.record_llm,
    )
    llm_clients = build_llm_clients(client_cfg)
    llm_planner = LLMPlanner(
        client=llm_clients.planner_client,
        model=llm_cfg["planner_model"],
    )
    llm_writer = LLMWriter(
        client=llm_clients.writer_client,
        model=llm_cfg["writer_model"],
    )
    validator_pass2_client = llm_clients.validator_client
    validator_pass2_model = llm_cfg["validator_model"]

    archetype_filter: tuple[str, ...] | None = None
    if args.archetypes:
        archetype_filter = tuple(s.strip() for s in args.archetypes.split(",") if s.strip())

    # Setup writer based on mode.
    if args.integrate:
        db, bucket = await _open_db_and_bucket()
        ingestion_writer = IngestionWriter(
            out_dir=out,
            mode="integrate",
            customer_id=profile.customer_id,
            bucket=bucket,
            db=db,
        )
    else:
        db = None
        ingestion_writer = IngestionWriter(out_dir=out, mode="local")

    try:
        emitted_docs: list = []
        specs_by_id: dict[str, ScenarioSpec] = {}
        async for spec, doc in run_scenarios(
            world,
            ownership,
            profile,
            time_window,
            archetype_filter=archetype_filter,
            scenario_limit=args.limit_scenarios,
            company_ctx=cc,
            planner=llm_planner,
            writer=llm_writer,
            validator_pass2_client=validator_pass2_client,
            validator_pass2_model=validator_pass2_model,
        ):
            await ingestion_writer.write(doc)
            emitted_docs.append(doc)
            specs_by_id.setdefault(spec.id, spec)
        await ingestion_writer.close()
    finally:
        if db is not None:
            await db.close()

    all_specs = list(specs_by_id.values())

    violations = validate_name_only(tuple(emitted_docs), world)

    finished_at = datetime.now(UTC)
    run_id = out.name

    archetypes_executed: dict[str, dict] = {}
    for doc in emitted_docs:
        slot = archetypes_executed.setdefault(
            doc.archetype, {"generated": 0, "dropped": 0}
        )
        slot["generated"] += 1
    # Plan 2 templated archetypes don't have a "requested" count distinct
    # from generated; Plan 3's LLM planner will emit per-scenario request
    # counts that distinguish these.

    totals = {
        "archetypes_executed": archetypes_executed,
        "totals": {
            "scenarios": len(specs_by_id),
            "documents": len(emitted_docs),
            "questions": sum(len(s.eval_questions) for s in all_specs),
        },
        "warnings_count": len(violations),
    }

    write_manifest(
        out,
        run_id=run_id,
        profile=profile,
        world=world,
        totals=totals,
        mode="integrate" if args.integrate else "local",
        started_at=started_at,
        finished_at=finished_at,
    )
    write_docs_index(out, emitted_docs)
    write_profile(out, profile)
    write_warnings(out, violations, [])
    write_scenarios(out, all_specs)
    write_questions_jsonl(out, all_specs, emitted_docs)
    (out / "world_model.json").write_text(_dumps(world))
    (out / "company_context.json").write_text(_dumps(cc))

    print(f"wrote {out}/manifest.json ({len(emitted_docs)} docs)", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "extract":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        return asyncio.run(_extract_async(profile, out))

    if args.cmd == "init":
        profile = load_profile(Path(args.profile))
        return asyncio.run(_init_async(profile))

    if args.cmd == "run":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        try:
            return asyncio.run(_run_async(profile, out, args))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4

    if args.cmd == "clean":
        try:
            return asyncio.run(_clean_async(args.customer))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4

    if args.cmd == "allow-seed":
        return asyncio.run(_allow_seed_async(args))

    if args.cmd == "seed":
        return asyncio.run(_seed_async(args))

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
