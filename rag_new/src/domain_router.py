import os
from typing import Optional

# Import lõi tìm kiếm để AI tự chấm điểm các domain
try:
    from .retrieval_service import retrieve_and_rerank
except ImportError:
    from retrieval_service import retrieve_and_rerank

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS_DIR = os.path.join(BASE_DIR, "data", "domains")

def get_available_domains() -> list[str]:
    """Tự động đọc danh sách các thư mục có trong kho"""
    if not os.path.exists(DOMAINS_DIR):
        return []
    return [d for d in os.listdir(DOMAINS_DIR) if os.path.isdir(os.path.join(DOMAINS_DIR, d))]

async def route_domain(user_query: str) -> Optional[str]:
    """
    [ĐỊNH TUYẾN THÔNG MINH - DÙNG RERANKER ĐỂ TỰ CHẤM ĐIỂM]
    Không cần từ khóa. Hệ thống sẽ thử tìm trong mọi domain, 
    domain nào có điểm khớp (score) cao nhất thì chọn domain đó.
    """
    domains = get_available_domains()
    
    if not domains:
        return "unknown"
        
    # Nếu chỉ có 1 domain thì không cần phải suy nghĩ
    if len(domains) == 1:
        return domains[0]

    best_domain = domains[0]
    max_score = -1.0

    print(f"--> [Auto-Router] Đang quét ngữ nghĩa tự động trên {len(domains)} lĩnh vực...")

    # Quét nhanh qua từng kho dữ liệu
    for d in domains:
        try:
            # Lấy thử 1 kết quả tốt nhất của từng kho để xem kho nào phù hợp nhất
            _, score, _ = retrieve_and_rerank(user_query, domain=d, top_k=1)
            
            # Ưu tiên cộng điểm nhẹ cho domain luật nếu điểm ngang ngửa, 
            # tránh trường hợp câu hỏi quá ngắn AI bị phân vân
            if ("thu_tuc" in d or "hanh_chinh" in d) and score > 0:
                score += 0.05 
                
            if score > max_score:
                max_score = score
                best_domain = d
        except Exception:
            continue
            
    return best_domain