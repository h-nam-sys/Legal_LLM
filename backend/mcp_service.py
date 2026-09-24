import os
import google.generativeai as genai
from dotenv import load_dotenv
import unicodedata
import re
import difflib

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

def normalize_text(text: str) -> str:
    """Removes Vietnamese accents, punctuation, and collapses whitespace."""
    if not text:
        return ""
    # Strip accents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove special characters and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

# Pre-calculate normalized vocab to save CPU cycles
NORMALIZED_VOCAB = [normalize_text(word) for word in MCP_TRIGGER_VOCAB]

def should_trigger_mcp(query: str) -> bool:
    """Checks if the user's latest query contains administrative vocabulary (Fuzzy & Normalized)."""
    norm_query = normalize_text(query)

    # 1. Exact Substring Match (handles missing accents like "thu tuc")
    for vocab in NORMALIZED_VOCAB:
        if vocab in norm_query:
            return True

    # 2. Fuzzy Match for Typos (e.g. "thuy tuc", "giay to")
    query_words = norm_query.split()
    for vocab in NORMALIZED_VOCAB:
        vocab_word_count = len(vocab.split())

        # Slide a window across the user's query matching the length of the vocab phrase
        for i in range(len(query_words) - vocab_word_count + 1):
            ngram = " ".join(query_words[i:i + vocab_word_count])

            # Compare similarity ratio
            similarity = difflib.SequenceMatcher(None, vocab, ngram).ratio()
            if similarity >= 0.85:  # 85% match threshold
                return True

    return False

async def execute_mcp_search(user_query: str, location: str = "Tăng Nhơn Phú") -> tuple[str, list[str]]:
    """Executes the external search using Gemini API and returns the final answer."""
    print(f"[MCP SERVICE] Triggering Gemini API for: '{user_query}' at '{location}'")

    if not GEMINI_API_KEY:
        return "Lỗi hệ thống: Quản trị viên chưa cấu hình khóa Gemini API.", ["Lỗi hệ thống"]

    try:
        # 1. Hardcode domain restriction
        domain_restriction = "site:dichvucong.gov.vn"

        # 2. Add geofencing context
        geo_context = f"tại {location}" if location else ""

        # 3. Combine into the strict final query
        strict_query = f"{user_query} {geo_context} {domain_restriction}"

        # gemini-1.5-flash or 2.5-flash are extremely fast for these lookups
        model = genai.GenerativeModel('gemini-2.5-flash')

        mcp_prompt = f"""Bạn là cán bộ hướng dẫn thủ tục hành chính tại {location}.
Người dân đang hỏi thủ tục nằm ngoài cơ sở dữ liệu nội bộ của hệ thống.

TỪ KHÓA TÌM KIẾM BẮT BUỘC: "{strict_query}"

YÊU CẦU BẮT BUỘC:
1. TRẢ LỜI CỰC KỲ NGẮN GỌN, VÀO THẲNG VẤN ĐỀ. Tuyệt đối KHÔNG chào hỏi rườm rà, KHÔNG dùng câu văn dài dòng.
2. Nêu trực tiếp nơi giải quyết thủ tục tại Tăng Nhơn Phú (Sở Giao thông, UBND...).
3. CHỈ liệt kê các giấy tờ cần chuẩn bị bằng dạng gạch đầu dòng (Checklist) siêu ngắn gọn.
4. CHỈ lấy thông tin từ dichvucong.gov.vn. KHÔNG giải thích dông dài."""

        # Use async generation so it doesn't block your FastAPI server
        response = await model.generate_content_async(mcp_prompt)

        return response.text, [f"Trợ lý Gemini ({domain_restriction})"]

    except Exception as e:
        return f"Hệ thống tìm kiếm ngoại tuyến đang bận: {str(e)}", ["Nguồn: Lỗi MCP"]
