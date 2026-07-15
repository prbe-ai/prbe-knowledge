from engine.ingest.handlers import registry
from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import SourceSystem
from kb.handlers.claude_code import ClaudeCodeConnector


def test_claude_code_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.CLAUDE_CODE)
    assert cls is ClaudeCodeConnector


def test_claude_code_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = ClaudeCodeConnector(ctx)
    assert c.source_system == SourceSystem.CLAUDE_CODE
