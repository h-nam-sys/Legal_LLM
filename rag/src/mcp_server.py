from fastmcp import FastMCP

# ============================================================
# 1. CREATE MCP SERVER
# ============================================================

mcp = FastMCP("Vietnamese Legal MCP Server")


# ============================================================
# 2. MOCK LAW TOOL
# ============================================================

@mcp.tool
def search_mock_laws(query: str) -> str:
    """
    Search mock Vietnamese legal documents.
    This is a simple hardcoded tool for testing MCP communication.
    """

    return """
Điều 33. Quyền của người lao động

1. Người lao động có quyền đơn phương chấm dứt hợp đồng lao động
theo quy định của pháp luật.

2. Khi đơn phương chấm dứt hợp đồng lao động, người lao động phải
thông báo trước cho người sử dụng lao động theo thời hạn quy định
của pháp luật.

3. Thời hạn báo trước phụ thuộc vào loại hợp đồng lao động và trường
hợp chấm dứt hợp đồng.

Đây là dữ liệu luật mẫu dùng để kiểm tra MCP Server.
"""


# ============================================================
# 3. RUN MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()