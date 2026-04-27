from shared.constants import SourceSystem
from shared.models import IntegrationToken


def test_integration_token_defaults_have_no_device() -> None:
    tok = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.SLACK,
        access_token="xoxb-fake",
    )
    assert tok.device_id is None
    assert tok.device_metadata is None


def test_integration_token_accepts_device_fields() -> None:
    tok = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="ignored-for-claude-code",
        webhook_secret="hash-of-device-token",
        device_id="dev-uuid-1",
        device_metadata={"os": "macos", "hostname": "mahits-mbp"},
    )
    assert tok.device_id == "dev-uuid-1"
    assert tok.device_metadata == {"os": "macos", "hostname": "mahits-mbp"}
