"""Thin deploy wrapper — the MCP server app now lives in engine.mcp.main.

Kept so `uvicorn services.mcp.main:app` (docker-compose and hosted charts)
keeps working unchanged.
"""

from engine.mcp.main import app

__all__ = ["app"]

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.mcp.main:app",
        host="0.0.0.0",
        port=8084,
        reload=False,
    )
