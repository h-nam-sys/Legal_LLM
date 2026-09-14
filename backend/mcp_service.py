import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API using your environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# The isolated vocabulary database for MCP triggers
MCP_TRIGGER_VOCAB = [
    "thủ tục", "giấy tờ", "hồ sơ", "luật", "quy định", "đăng ký",
    "xin cấp", "ủy ban", "phường", "công an", "công chứng",
    "đóng dấu", "pháp lý", "chứng nhận", "bản sao", "lệ phí", "nhà nước"
]

def should_trigger_mcp(query: str) -> bool:
    """Checks if the user's latest query contains administrative vocabulary."""
    query_lower = query.lower()
    return any(word in query_lower for word in MCP_TRIGGER_VOCAB)

async def execute_mcp_search(query: str) -> tuple[str, list[str]]:
    """Executes the external search using Gemini API and returns the final answer."""
    print(f"[MCP SERVICE] Triggering Gemini API for: '{query}'")

    if not GEMINI_API_KEY:
        return "Lỗi hệ thống: Quản trị viên chưa cấu hình khóa Gemini API.", ["Nguồn: Lỗi hệ thống"]

    try:
        # gemini-1.5-flash is extremely fast and cost-effective for these lookups
        model = genai.GenerativeModel('gemini-2.5-flash')

        mcp_prompt = f"""Bạn là cán bộ hướng dẫn thủ tục hành chính tại Việt Nam.
Người dân đang hỏi thủ tục nằm ngoài cơ sở dữ liệu nội bộ của hệ thống.

CÂU HỎI CỦA NGƯỜI DÂN: "{query}"

YÊU CẦU:
1. Dựa vào kiến thức pháp luật hiện hành, hãy trả lời ngắn gọn, lịch sự.
2. Liệt kê các giấy tờ cơ bản cần thiết (nếu có).
3. KHÔNG dùng văn phong luật sư dài dòng, KHÔNG dùng ngoặc vuông."""

        # Use async generation so it doesn't block your FastAPI server
        response = await model.generate_content_async(mcp_prompt)

        return response.text, ["Nguồn: Trợ lý thông minh (Gemini)"]

    except Exception as e:
        return f"Hệ thống tìm kiếm ngoại tuyến đang bận: {str(e)}", ["Nguồn: Lỗi MCP"]
