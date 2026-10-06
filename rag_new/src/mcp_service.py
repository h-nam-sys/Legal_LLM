import json

DEFAULT_MISSING_TEXT = "Hệ thống hiện chưa có dữ liệu chi tiết cho mục này."

async def invoke_mcp_as_rag_payload(user_query: str, domain: str, llm_url: str = "") -> dict:
    """
    [BACKEND DEV MODE] Trả về JSON giả lập (Mock Data) không cần gọi LLM.
    Đại diện cho việc hệ thống đã ra ngoài tìm dữ liệu và đóng gói thành công.
    """
    print(f"      [Mock MCP] Đang gọi API bên ngoài cho câu hỏi '{user_query}'...")
    
    mock_payload = {
        "Nguồn dữ liệu": "Web Search / External API (Giả lập)",
        "Lĩnh vực": domain,
        "Dữ liệu thu thập được": f"Đây là kết quả tìm kiếm từ bên ngoài cho câu hỏi: '{user_query}'",
        "Trạng thái": "Thành công",
        "Ghi chú bổ sung": DEFAULT_MISSING_TEXT
    }
    
    return mock_payload