"""Provider-agnostic structured output adapter.

generate_typed() calls client.generate_structured() and validates the
result through Pydantic, raising StructuredOutputValidationError on
schema mismatch so callers have a single stable exception to catch.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from scripts.synth.llm.base import LlmClientProtocol, LlmRequest


class StructuredOutputValidationError(Exception):
    """Raised when the LLM response dict fails Pydantic validation."""


async def generate_typed[T: BaseModel](
    client: LlmClientProtocol,
    req: LlmRequest,
    schema: type[T],
) -> T:
    """Call client.generate_structured and validate the result as a Pydantic model.

    Args:
        client: Any LlmClientProtocol implementor (Anthropic, Gemini, Mock).
        req: The LlmRequest to send.
        schema: The Pydantic BaseModel subclass to validate against.

    Returns:
        A validated instance of *schema*.

    Raises:
        StructuredOutputValidationError: If the raw dict does not match *schema*.
    """
    raw = await client.generate_structured(req, schema)
    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        raise StructuredOutputValidationError(
            f"LLM response failed validation for {schema.__name__}: {exc}"
        ) from exc
