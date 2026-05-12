"""Eval harness for directed-phrase generation: Haiku 4.5 vs Gemini 3.1 Pro
vs Gemini 3.1 Flash Lite vs Gemini 3 Flash.

Runs each model on 5 fixtures (2 prod + 3 synthetic), captures phrases,
tokens, latency, then judges quality with Opus 4.7. Embeds phrases via
OpenAI to compute distinctness + retrieval-fitness. Writes a markdown
report to ~/.gstack/projects/prbe-knowledge/eval-directed-phrases-<ts>.md
plus a JSONL with per-call detail.

Manual harness, NOT in CI. The run hits 4 paid LLM APIs + OpenAI embeddings
(~$0.50, ~5 min wall time). Re-run when you change the directed-phrase
prompt or want to evaluate a new model id; pin the constant in
shared/constants.py:DIRECTED_PHRASES_MODEL based on the report.

Run:
    cd /path/to/prbe-knowledge
    uv run python scripts/eval_directed_phrases.py

Acceptance gates for a Gemini 3 Flash candidate (current default):
    avg specificity        >= 8.0   (baseline 8.6)
    avg retrieval-fit      >= 8.0   (baseline 8.2)
    avg distinctness       >= 7.0   (baseline 7.6)
    cost per call          <= $0.001 (baseline $0.00054)

Baseline report: ~/.gstack/projects/prbe-knowledge/eval-directed-phrases-20260509-204625.md
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# ruff: noqa: E402  (intentional: dynamic .env load before SDK imports)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import google.genai as genai  # type: ignore[import-untyped]
from anthropic import AsyncAnthropic
from google.genai import types as genai_types  # type: ignore[import-untyped]
from openai import AsyncOpenAI

from scripts.eval_data.fixtures import FIXTURES, Fixture

# ---- Models under test ----------------------------------------------------

HAIKU = "claude-haiku-4-5-20251001"
GEMINI_PRO = "gemini-3.1-pro-preview"
GEMINI_FLASH_LITE = "gemini-3.1-flash-lite"
GEMINI_FLASH = "gemini-3-flash-preview"

JUDGE_MODEL = "claude-opus-4-7"  # user-selected: Opus 4.7 as judge

# Pricing per 1M tokens (input, output). Verified against public pricing pages
# at the time of run; bump in one place if pricing moves.
PRICING: dict[str, tuple[float, float]] = {
    HAIKU:             (1.00,  5.00),
    GEMINI_PRO:        (1.25, 10.00),
    GEMINI_FLASH_LITE: (0.10,  0.40),
    GEMINI_FLASH:      (0.30,  2.50),
    JUDGE_MODEL:       (15.0, 75.0),
}

# ---- The actual prompt ----------------------------------------------------
# Mirrors services/synthesis/prompts.py:_directed_system verbatim.

SYSTEM_PROMPT = (
    "You write retrieval trigger phrases for a wiki page. The phrases will "
    "be embedded and matched against future user queries; a semantic match "
    "boosts this page in retrieval ranking.\n\n"
    "Goal: when an engineer asks about the SPECIFIC problem this page "
    "addresses (in their own words, phrased as a symptom or situation), "
    "the page should surface — even if their query doesn't share words "
    "with the page body.\n\n"
    "Rules:\n"
    "  - 5-10 phrases. Each 5-12 tokens.\n"
    "  - Specific symptoms over generic IT terms ('deploy timeout on "
    "    fly machine after 5 minutes' beats 'deployment problem').\n"
    "  - Phrase the situation as a user would describe it.\n"
    "  - Do not include the page title verbatim.\n"
    "  - Do not include phrases that would match unrelated runbooks.\n"
    "  - Cover the page's distinct retrieval angles (different symptoms "
    "    or situations the page addresses).\n"
)

USER_TEMPLATE = "Page title:\n{title}\n\nPage body:\n<body>\n{body}\n</body>\n"

TOOL_NAME = "record_directed_phrases"
TOOL_DESCRIPTION = (
    "Emit 5-10 short trigger phrases the page should be retrievable for. "
    "Phrases describe SPECIFIC SYMPTOMS or SITUATIONS, not generic IT terms."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "5-10 trigger phrases, each 5-12 tokens, describing problems "
                "or situations this page should be retrieved for."
            ),
        }
    },
    "required": ["phrases"],
}


# ---- Result records --------------------------------------------------------


@dataclass
class GenerationResult:
    model: str
    fixture_id: str
    phrases: list[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None


@dataclass
class JudgedResult:
    model: str
    fixture_id: str
    phrases: list[str]
    specificity: int           # 1-10
    distinctness_judge: int    # 1-10 (judge's view)
    retrieval_fitness: int     # 1-10
    rationale: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


@dataclass
class FixtureMetrics:
    model: str
    fixture_id: str
    # Adherence (rule-based, no LLM)
    phrase_count: int
    in_count_range: bool       # 5 <= n <= 10
    pct_in_token_range: float  # % phrases with 5-12 word tokens
    title_verbatim_count: int  # how many phrases include title verbatim
    # Distinctness from embeddings
    avg_pairwise_cosine: float  # lower = more distinct
    max_pairwise_cosine: float  # worst-case overlap
    # Retrieval-fitness from engineer-query embeddings
    best_engineer_query_match: float  # max cosine sim of any phrase to
                                       # any engineer query (higher = better)
    avg_engineer_query_match: float    # avg of (max sim per query)
    # Cost + latency
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    # Judge scores (filled later)
    specificity: int = 0
    distinctness_judge: int = 0
    retrieval_fitness_judge: int = 0
    judge_rationale: str = ""


# ---- Generation runners ----------------------------------------------------


async def run_anthropic(model: str, fixture: Fixture) -> GenerationResult:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user = USER_TEMPLATE.format(title=fixture.title, body=fixture.body)
    t0 = time.perf_counter()
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT}],
            tools=[{
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "input_schema": JSON_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        phrases: list[str] = []
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" \
               and getattr(block, "name", "") == TOOL_NAME:
                payload = getattr(block, "input", {}) or {}
                raw = payload.get("phrases", [])
                phrases = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
                break
        return GenerationResult(
            model=model,
            fixture_id=fixture.fixture_id,
            phrases=phrases,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return GenerationResult(
            model=model, fixture_id=fixture.fixture_id, phrases=[],
            input_tokens=0, output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_gemini(model: str, fixture: Fixture) -> GenerationResult:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    user = USER_TEMPLATE.format(title=fixture.title, body=fixture.body)
    contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=user)])]
    # Gemini 3.x defaults to thinking-on; reasoning tokens consume
    # max_output_tokens silently and truncate the JSON answer mid-string.
    # Per-model handling:
    #   - Pro: thinking is mandatory (rejects budget=0). Give it slack
    #     (4096 thinking + 2048 answer headroom).
    #   - Flash / Flash Lite: thinking_budget=0 disables it; 2048 output
    #     is plenty for 5-10 phrases.
    if "pro" in model:
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=JSON_SCHEMA,
            max_output_tokens=8192,
            temperature=0.0,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=4096),
        )
    else:
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=JSON_SCHEMA,
            max_output_tokens=2048,
            temperature=0.0,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )
    t0 = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=model, contents=contents, config=config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (resp.text or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return GenerationResult(
                model=model, fixture_id=fixture.fixture_id, phrases=[],
                input_tokens=resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0,
                output_tokens=resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0,
                latency_ms=latency_ms,
                error=f"json_parse_failed: {text[:200]!r}",
            )
        raw = parsed.get("phrases", [])
        phrases = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
        usage = resp.usage_metadata
        return GenerationResult(
            model=model,
            fixture_id=fixture.fixture_id,
            phrases=phrases,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return GenerationResult(
            model=model, fixture_id=fixture.fixture_id, phrases=[],
            input_tokens=0, output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---- Embedding helper ------------------------------------------------------


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed via OpenAI (matches what the production directed retriever
    uses). Returns parallel list of vectors.
    """
    if not texts:
        return []
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


# ---- Adherence + distinctness + retrieval-fitness computation -------------


def compute_metrics(
    gen: GenerationResult,
    fixture: Fixture,
    phrase_vecs: list[list[float]],
    engineer_query_vecs: list[list[float]],
) -> FixtureMetrics:
    phrases = gen.phrases
    n = len(phrases)

    # Adherence
    in_count = 5 <= n <= 10
    token_counts = [len(p.split()) for p in phrases]
    in_token = sum(1 for t in token_counts if 5 <= t <= 12)
    pct_in_token = (in_token / n * 100) if n else 0.0
    title_lower = fixture.title.lower()
    title_verb = sum(1 for p in phrases if title_lower in p.lower())

    # Distinctness — pairwise cosine on emitted phrase embeddings
    if len(phrase_vecs) >= 2:
        pairs = [
            _cosine(phrase_vecs[i], phrase_vecs[j])
            for i in range(len(phrase_vecs))
            for j in range(i + 1, len(phrase_vecs))
        ]
        avg_pair = statistics.mean(pairs)
        max_pair = max(pairs)
    else:
        avg_pair = 0.0
        max_pair = 0.0

    # Retrieval-fitness — for each engineer query, find the BEST phrase
    # similarity, then aggregate.
    if phrase_vecs and engineer_query_vecs:
        best_per_query = []
        for qv in engineer_query_vecs:
            sims = [_cosine(qv, pv) for pv in phrase_vecs]
            best_per_query.append(max(sims) if sims else 0.0)
        best_eng = max(best_per_query)
        avg_eng = statistics.mean(best_per_query)
    else:
        best_eng = 0.0
        avg_eng = 0.0

    inp_cost, out_cost = PRICING[gen.model]
    cost = (gen.input_tokens / 1_000_000) * inp_cost + (gen.output_tokens / 1_000_000) * out_cost

    return FixtureMetrics(
        model=gen.model,
        fixture_id=gen.fixture_id,
        phrase_count=n,
        in_count_range=in_count,
        pct_in_token_range=pct_in_token,
        title_verbatim_count=title_verb,
        avg_pairwise_cosine=avg_pair,
        max_pairwise_cosine=max_pair,
        best_engineer_query_match=best_eng,
        avg_engineer_query_match=avg_eng,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        cost_usd=cost,
        latency_ms=gen.latency_ms,
    )


# ---- Judge -----------------------------------------------------------------


JUDGE_SYSTEM = (
    "You are evaluating LLM-generated retrieval trigger phrases for a wiki "
    "page. The phrases will be embedded and matched against future user "
    "queries; a semantic match boosts the page in retrieval ranking.\n\n"
    "Score on three dimensions, 1-10 each:\n"
    "  - SPECIFICITY: how concrete and symptom-shaped are the phrases? "
    "    'deploy timing out after 5 min on fly machine' is high; "
    "    'deployment problem' is low.\n"
    "  - DISTINCTNESS: how different from each other? Near-paraphrases score low.\n"
    "  - RETRIEVAL_FITNESS: how likely a real engineer's symptom-style "
    "    query would match these phrases. Generic IT-terms score low because "
    "    they'd over-match unrelated pages.\n\n"
    "Provide a one-paragraph rationale citing specific phrases."
)

JUDGE_TOOL = {
    "name": "record_judgment",
    "description": "Record numeric scores and a rationale for the trigger phrases.",
    "input_schema": {
        "type": "object",
        "properties": {
            "specificity": {"type": "integer", "minimum": 1, "maximum": 10},
            "distinctness": {"type": "integer", "minimum": 1, "maximum": 10},
            "retrieval_fitness": {"type": "integer", "minimum": 1, "maximum": 10},
            "rationale": {"type": "string"},
        },
        "required": ["specificity", "distinctness", "retrieval_fitness", "rationale"],
    },
}


async def judge(gen: GenerationResult, fixture: Fixture) -> JudgedResult:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not gen.phrases:
        return JudgedResult(
            model=gen.model, fixture_id=gen.fixture_id, phrases=[],
            specificity=0, distinctness_judge=0, retrieval_fitness=0,
            rationale=f"Skipped: model returned no phrases (error={gen.error}).",
        )
    user = (
        f"Page title:\n{fixture.title}\n\n"
        f"Page body (first 2000 chars):\n{fixture.body[:2000]}\n\n"
        f"Phrases to score:\n"
        + "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(gen.phrases))
    )
    resp = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": JUDGE_SYSTEM}],
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_judgment"},
        messages=[{"role": "user", "content": user}],
    )
    payload = {}
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" \
           and getattr(block, "name", "") == "record_judgment":
            payload = getattr(block, "input", {}) or {}
            break
    return JudgedResult(
        model=gen.model,
        fixture_id=gen.fixture_id,
        phrases=gen.phrases,
        specificity=int(payload.get("specificity", 0)),
        distinctness_judge=int(payload.get("distinctness", 0)),
        retrieval_fitness=int(payload.get("retrieval_fitness", 0)),
        rationale=str(payload.get("rationale", "")),
        judge_input_tokens=resp.usage.input_tokens,
        judge_output_tokens=resp.usage.output_tokens,
    )


# ---- Orchestration ---------------------------------------------------------


async def run_one_model_fixture(
    model: str, fixture: Fixture,
) -> tuple[GenerationResult, JudgedResult, FixtureMetrics]:
    if model == HAIKU:
        gen = await run_anthropic(model, fixture)
    else:
        gen = await run_gemini(model, fixture)

    # Embed phrases + engineer queries.
    phrase_vecs = await embed_batch(gen.phrases)
    eng_vecs = await embed_batch(fixture.engineer_queries)

    metrics = compute_metrics(gen, fixture, phrase_vecs, eng_vecs)
    judgment = await judge(gen, fixture)
    metrics.specificity = judgment.specificity
    metrics.distinctness_judge = judgment.distinctness_judge
    metrics.retrieval_fitness_judge = judgment.retrieval_fitness
    metrics.judge_rationale = judgment.rationale
    return gen, judgment, metrics


async def main() -> None:
    models = [HAIKU, GEMINI_PRO, GEMINI_FLASH_LITE, GEMINI_FLASH]
    all_metrics: list[FixtureMetrics] = []
    all_gens: list[GenerationResult] = []
    all_judgments: list[JudgedResult] = []

    print(f"Eval start: {len(models)} models * {len(FIXTURES)} fixtures = {len(models) * len(FIXTURES)} runs", flush=True)

    for fixture in FIXTURES:
        print(f"\n[fixture] {fixture.fixture_id} ({fixture.title})", flush=True)
        for model in models:
            print(f"  [{model}] running... ", end="", flush=True)
            gen, judgment, metrics = await run_one_model_fixture(model, fixture)
            all_gens.append(gen)
            all_judgments.append(judgment)
            all_metrics.append(metrics)
            if gen.error:
                print(f"ERROR {gen.error}", flush=True)
            else:
                print(
                    f"phrases={metrics.phrase_count} "
                    f"specificity={metrics.specificity} "
                    f"retrieval_fit={metrics.retrieval_fitness_judge} "
                    f"cost=${metrics.cost_usd:.4f} "
                    f"lat={metrics.latency_ms:.0f}ms",
                    flush=True,
                )

    # Write report
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path.home() / ".gstack" / "projects" / "prbe-knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval-directed-phrases-{ts}.md"
    out_path.write_text(_render_report(all_metrics, all_gens, all_judgments))

    # Also dump JSONL for downstream tooling.
    raw_path = out_dir / f"eval-directed-phrases-{ts}.jsonl"
    with raw_path.open("w") as f:
        for m, g, j in zip(all_metrics, all_gens, all_judgments, strict=True):
            f.write(json.dumps({
                "metrics": asdict(m),
                "generation": asdict(g),
                "judgment": asdict(j),
            }) + "\n")

    print(f"\n\nReport: {out_path}", flush=True)
    print(f"Raw: {raw_path}", flush=True)


# ---- Reporting -------------------------------------------------------------


def _agg(metrics: list[FixtureMetrics], model: str, attr: str) -> float:
    vals = [getattr(m, attr) for m in metrics if m.model == model]
    return statistics.mean(vals) if vals else 0.0


def _sum(metrics: list[FixtureMetrics], model: str, attr: str) -> float:
    return sum(getattr(m, attr) for m in metrics if m.model == model)


def _render_report(
    metrics: list[FixtureMetrics],
    gens: list[GenerationResult],
    judgments: list[JudgedResult],
) -> str:
    models = [HAIKU, GEMINI_PRO, GEMINI_FLASH_LITE, GEMINI_FLASH]
    lines: list[str] = []
    lines.append(f"# Directed-phrase generation eval — {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"Fixtures: {len(FIXTURES)} (2 prod + 3 synthetic)")
    lines.append(f"Judge: {JUDGE_MODEL}")
    lines.append("Embedder for distinctness/retrieval-fitness: text-embedding-3-large")
    lines.append("")

    # ---- Aggregate table ----
    lines.append("## Per-model aggregates")
    lines.append("")
    lines.append("| Model | Avg specificity | Avg retrieval-fit | Avg distinctness (judge) | Avg pairwise cos (lower=better) | Avg engineer-query match | Avg phrase count | Latency p50 ms | Total cost ($) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        lat_vals = sorted(x.latency_ms for x in metrics if x.model == m)
        p50 = lat_vals[len(lat_vals) // 2] if lat_vals else 0.0
        lines.append(
            f"| `{m}` | {_agg(metrics, m, 'specificity'):.1f} | "
            f"{_agg(metrics, m, 'retrieval_fitness_judge'):.1f} | "
            f"{_agg(metrics, m, 'distinctness_judge'):.1f} | "
            f"{_agg(metrics, m, 'avg_pairwise_cosine'):.3f} | "
            f"{_agg(metrics, m, 'avg_engineer_query_match'):.3f} | "
            f"{_agg(metrics, m, 'phrase_count'):.1f} | "
            f"{p50:.0f} | "
            f"{_sum(metrics, m, 'cost_usd'):.4f} |"
        )
    lines.append("")

    # ---- Cost-per-call estimate ----
    lines.append("## Estimated cost per page synthesis")
    lines.append("")
    lines.append("| Model | Avg input tokens | Avg output tokens | Cost / call |")
    lines.append("|---|---|---|---|")
    for m in models:
        avg_in = _agg(metrics, m, "input_tokens")
        avg_out = _agg(metrics, m, "output_tokens")
        avg_cost = _agg(metrics, m, "cost_usd")
        lines.append(f"| `{m}` | {avg_in:.0f} | {avg_out:.0f} | ${avg_cost:.5f} |")
    lines.append("")

    # ---- Per-fixture detail ----
    lines.append("## Per-fixture detail")
    lines.append("")
    for fixture in FIXTURES:
        lines.append(f"### {fixture.fixture_id} — {fixture.title}")
        lines.append("")
        lines.append(f"_{fixture.note}_")
        lines.append("")
        lines.append("Engineer queries (used for retrieval-fitness):")
        for q in fixture.engineer_queries:
            lines.append(f"  - `{q}`")
        lines.append("")
        for m in models:
            entry = next((x for x in metrics if x.model == m and x.fixture_id == fixture.fixture_id), None)
            gen = next((x for x in gens if x.model == m and x.fixture_id == fixture.fixture_id), None)
            if entry is None or gen is None:
                continue
            lines.append(f"#### `{m}`")
            lines.append("")
            if gen.error:
                lines.append(f"ERROR: `{gen.error}`")
                lines.append("")
                continue
            lines.append(
                f"specificity {entry.specificity}/10 · "
                f"retrieval-fit {entry.retrieval_fitness_judge}/10 · "
                f"distinctness {entry.distinctness_judge}/10 · "
                f"avg pair cos {entry.avg_pairwise_cosine:.3f} · "
                f"max query match {entry.best_engineer_query_match:.3f} · "
                f"cost ${entry.cost_usd:.5f} · {entry.latency_ms:.0f}ms"
            )
            lines.append("")
            lines.append("Phrases:")
            for p in gen.phrases:
                lines.append(f"  - `{p}`")
            lines.append("")
            if entry.judge_rationale:
                lines.append(f"Judge: {entry.judge_rationale}")
                lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
