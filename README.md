# llmgui
Over simple Gradio interface for llama.cpp server

## Start llama.cpp 

llama-server.exe --jinja --ctx-size 40960 -m g:\bidule_llama\models\Magistral-Small-2506-Q4_K_M.gguf -ngl 99

# Todo :
https://gofastmcp.com/getting-started/quickstart

from fastmcp import FastMCP

mcp = FastMCP("Utility MCP Server")

# fastmcp run mcp-server.py:mcp --transport http --port 8000

@mcp.tool
def get_weather(location: str) -> str:
    """Return weather information for the requested location (stub)."""
    return f"Weather lookup not yet implemented for {location}."


@mcp.tool
def get_date_time(timezone: str | None = None) -> str:
    """Return the current date/time, optionally for a timezone (stub)."""
    zone = timezone or "system default"
    return f"Date/time lookup not yet implemented for {zone}."


@mcp.tool
def get_stock_price(symbol: str) -> str:
    """Return the current market price for the given stock symbol (stub)."""
    return f"Stock price lookup not yet implemented for {symbol}."


if __name__ == "__main__":
    mcp.run()
