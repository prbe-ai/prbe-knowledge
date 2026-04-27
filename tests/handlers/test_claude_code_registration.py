from services.ingestion.handlers import registry
from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.constants import SourceSystem


def test_claude_code_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.CLAUDE_CODE)
    assert cls is ClaudeCodeConnector


def test_claude_code_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = ClaudeCodeConnector(ctx)
    assert c.source_system == SourceSystem.CLAUDE_CODE
