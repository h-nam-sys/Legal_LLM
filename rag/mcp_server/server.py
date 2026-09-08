from datetime import datetime

from mcp.server import MCPServer


mcp = MCPServer("Legal RAG MCP Server")


@mcp.tool()
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def get_random_law() -> str:
    """Return a mock legal text for testing."""
    return (
        "Bộ luật Lao động - Điều 35: "
        "Người lao động có quyền đơn phương chấm dứt "
        "hợp đồng lao động theo quy định của pháp luật."
    )


if __name__ == "__main__":
    mcp.run()