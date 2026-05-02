"""LLMWriter — produces plain text for a single DocSpec via one LLM call.

Persona-scoped prior context: prior_emitted_docs is filtered to only include
docs whose occurred_at is strictly before the current persona's first emission
in the scenario timeline (determined by the DocSpec's occurred_at).

Allowlist injection: every prompt includes the explicit list of allowed
service/people/channel names. Writers must not invent names outside this list,
except for incidental third-party SaaS (Stripe, AWS, Datadog, etc.).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import DocSpec, ScenarioSpec, Source
from scripts.synth.llm.base import LlmClientProtocol, LlmRequest
from scripts.synth.output.base import SynthDoc
from shared.logging import get_logger

if TYPE_CHECKING:
    from scripts.synth.company_context import CompanyContext
    from scripts.synth.world_model import WorldModel

log = get_logger(__name__)

# Default location of writer prompt templates relative to this file.
_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _filter_prior_for_persona(
    prior_docs: tuple[SynthDoc, ...],
    spec: ScenarioSpec,
    source: Source,
    emission_index: int,
) -> tuple[SynthDoc, ...]:
    """Return prior docs visible to the current persona at the time of their emission.

    The current persona is the one listed first in the DocSpec for this
    (source, emission_index) combination. Visibility cutoff is the occurred_at
    of that DocSpec — only docs strictly before that timestamp are included.
    """
    # Find the DocSpec being written
    target_doc: DocSpec | None = None
    count = 0
    for ds in spec.doc_specs:
        if ds.source == source:
            if count == emission_index:
                target_doc = ds
                break
            count += 1

    if target_doc is None or not target_doc.personas:
        # Can't determine persona; return nothing to be safe
        return ()

    cutoff_ts: datetime = target_doc.occurred_at
    return tuple(d for d in prior_docs if d.occurred_at < cutoff_ts)


class LLMWriter:
    def __init__(
        self,
        client: LlmClientProtocol,
        model: str,
        prompts_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR

    async def write(
        self,
        spec: ScenarioSpec,
        source: Source,
        emission_index: int,
        prior_emitted_docs: tuple[SynthDoc, ...],
        world: WorldModel,
        company_ctx: CompanyContext,
    ) -> str:
        """Single LLM call producing plain text for a single DocSpec.

        prior_emitted_docs is filtered to the persona's view before injection.
        """
        source_val = source.value if hasattr(source, "value") else str(source)
        template_path = self._prompts_dir / f"writer_{source_val}.txt"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Writer prompt template not found: {template_path}. "
                f"Create it or run with --record-llm to generate fixtures."
            )

        template = template_path.read_text(encoding="utf-8")

        # Persona-scoped prior context
        visible_docs = _filter_prior_for_persona(prior_emitted_docs, spec, source, emission_index)
        persona_view = (
            "\n---\n".join(
                f"[{d.source} | {d.occurred_at.isoformat()}]\n{d.text}"
                for d in visible_docs
            )
            if visible_docs
            else "(no prior context)"
        )

        # Allowlists
        allowed_services = ", ".join(
            sorted({s.qualified for s in world.services} | {s.name for s in world.services})
        )
        allowed_people = ", ".join(
            sorted(
                {p.display_name for p in world.people if p.display_name}
                | {p.gh_username for p in world.people if p.gh_username}
            )
        )
        allowed_channels = ", ".join(sorted(ch.name for ch in world.channels))

        # Current emission description
        target_doc: DocSpec | None = None
        count = 0
        for ds in spec.doc_specs:
            if ds.source == source:
                if count == emission_index:
                    target_doc = ds
                    break
                count += 1

        current_emission = (
            f"source={source_val}, channel={target_doc.channel or 'N/A'}, "
            f"personas={', '.join(target_doc.personas)}"
            if target_doc
            else f"source={source_val}, emission #{emission_index}"
        )

        scenario_summary = (
            f"Scenario: {getattr(spec, 'title', spec.id)}\n"
            f"Summary: {getattr(spec, 'summary', '')}\n"
            f"Cast: {', '.join(spec.cast)}\n"
            f"Services: {', '.join(spec.affected_services)}"
        )

        prompt = template.format(
            scenario_summary=scenario_summary,
            persona_view=persona_view,
            allowed_services=allowed_services,
            allowed_people=allowed_people,
            allowed_channels=allowed_channels,
            current_emission=current_emission,
            instance_ts=spec.instance_ts.isoformat(),
        )

        req = LlmRequest(
            model=self._model,
            system=(
                "You are a synthetic document generator. Write realistic, coherent content "
                "for the specified source. Use ONLY names from the allowlists provided. "
                "Do not invent company-internal names. Third-party SaaS (Stripe, AWS, "
                "Datadog, PagerDuty) are permitted incidentally."
            ),
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
        )

        log.debug(
            "llm_writer.write",
            source=source_val,
            emission_index=emission_index,
            scenario_id=spec.id,
            model=self._model,
        )

        response = await self._client.generate(req)
        return response.text
