import pytest

from shared.constants import SourceSystem
from shared.db import get_pool
from shared.tokens import load_token


@pytest.mark.asyncio
async def test_load_token_ignores_device_rows(live_db: None) -> None:
    """A device-scoped row for the same (customer, source) must NOT be returned
    by load_token — load_token is the singleton helper."""
    customer = "tokens-test-cust"
    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) VALUES ($1, 'tt', 'tt-hash') ON CONFLICT DO NOTHING",
            customer,
        )
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = $1", customer)
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status,
                 device_id, device_metadata)
            VALUES ($1, 'claude_code', 'fake-enc', 'active', 'dev-1', '{}')
            """,
            customer,
        )

    got = await load_token(customer, SourceSystem.CLAUDE_CODE)
    assert got is None

    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = $1", customer)
