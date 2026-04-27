import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.constants import SourceSystem
from shared.models import IntegrationToken


@pytest.mark.asyncio
async def test_identify_workspaces_returns_one_ref_per_device() -> None:
    c = ClaudeCodeConnector(make_default_context())
    tok = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="ignored",
        webhook_secret="hash",
        device_id="dev-uuid-1",
        device_metadata={"hostname": "mahits-mbp", "os": "macos"},
    )
    refs = await c.identify_workspaces(tok)
    assert len(refs) == 1
    assert refs[0].external_id == "dev-uuid-1"
    assert refs[0].external_name == "mahits-mbp"


@pytest.mark.asyncio
async def test_identify_workspaces_requires_device_id() -> None:
    c = ClaudeCodeConnector(make_default_context())
    tok = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="ignored",
    )
    with pytest.raises(ValueError):
        await c.identify_workspaces(tok)
