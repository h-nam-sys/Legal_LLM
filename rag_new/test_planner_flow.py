import asyncio
import os
import sys
import json
import pandas as pd
from sqlalchemy import create_engine, text

# 1. Khai báo đường dẫn gốc để Python nhận diện được package 'src'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)

# 2. Import từ các module của hệ thống (BẮT BUỘC CÓ TIỀN TỐ 'src.')
from src.domain_router import route_domain
from src.semantic_planner import build_tasks
from src.context_manager import ContextManager
from src.query_pipeline import fetch_record_from_sqlite, sanitize_value, SQLITE_DB_PATH
from src.retrieval_service import retrieve_and_rerank

# URL kết nối SQLite đồng bộ dùng cho pandas
SYNC_DB_URL = f"sqlite:///{SQLITE_DB_PATH}"

def load_procedures_from_sqlite(domain: str) -> pd.DataFrame:
    """Tự động kéo danh sách ID và Tên bài viết từ SQLite tùy theo cấu trúc của từng Domain"""
    try:
        engine = create_engine(SYNC_DB_URL)
        
        # [KHO LUẬT PHÁP]
        if "thu_tuc" in domain or "hanh_chinh" in domain:
            query = text(f'''
                SELECT 
                    row_id AS procedure_id, 
                    "Tên thủ tục hành chính" AS procedure_name, 
                    "{domain}" AS domain 
                FROM "{domain}"
            ''')
            
        # [KHO TÀI CHÍNH]
        elif domain == "fin":
            query = text(f'''
                SELECT 
                    vnfinsqa_id AS procedure_id, 
                    question AS procedure_name, 
                    "{domain}" AS domain 
                FROM "{domain}"
            ''')
            
        # [KHO IT / DATABASE]
        elif domain == "it_database":
            query = text(f'''
                SELECT 
                    row_id AS procedure_id, 
                    question AS procedure_name, 
                    "{domain}" AS domain 
                FROM "{domain}"
            ''')
            
        # [DOMAIN KHÁC] Trả về DataFrame rỗng để ép hệ thống dùng Vector Search
        else:
            return pd.DataFrame(columns=["procedure_id", "procedure_name", "domain"])

        df = pd.read_sql(query, engine)
        df['procedure_id'] = df['procedure_id'].astype(str)
        return df
        
    except Exception as e:
        print(f"[!] Lỗi load data từ SQLite bảng {domain}: {e}")
        return pd.DataFrame(columns=["procedure_id", "procedure_name", "domain"])


async def main():
    print("=====================================================================")
    print("   HỆ THỐNG RAG ĐA LĨNH VỰC (AUTO-ROUTING + SEMANTIC PLANNER)  ")
    print("   Gõ 'exit' hoặc 'quit' để thoát")
    print("=====================================================================\n")

    # Khởi tạo bộ nhớ quản lý ngữ cảnh đa lượt
    ctx = ContextManager()

    while True:
        try:
            user_query = input("\n[?] Bạn: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                break

            # 1. ĐỊNH TUYẾN DOMAIN (Auto-Router)
            domain = await route_domain(user_query)
            if domain == "unknown":
                print("--> [Router] Lĩnh vực lạ. Hệ thống chưa có dữ liệu mảng này.")
                continue
            print(f"--> [Router] Đã chốt định tuyến câu hỏi vào kho dữ liệu: '{domain}'")

            # 2. NẠP CHỦ ĐỀ TỪ DATABASE
            procedures_df = load_procedures_from_sqlite(domain)
            if procedures_df.empty and ("thu_tuc" in domain or "hanh_chinh" in domain):
                print(f"--> [Cảnh báo] Bảng {domain} trống. Vui lòng ingest dữ liệu trước.")
                continue

            # 3. LẬP KẾ HOẠCH (SEMANTIC PLANNER)
            raw_tasks = build_tasks(user_query, procedures_df)

            # 4. ÁP DỤNG NGỮ CẢNH (CONTEXT MANAGER)
            resolved_tasks = ctx.resolve(raw_tasks, user_query)
            primary_task = resolved_tasks[0] if resolved_tasks else None

            record_id = None
            score = 0.0
            source_type = ""
            intent = primary_task.get("intent") if primary_task else "general_information"

            # 5. XỬ LÝ TRUY XUẤT (RETRIEVAL)
            if primary_task and primary_task.get("procedure_id") and primary_task.get("status") == "ready":
                # [A] Trúng đích bằng luật Planner (chuyên dùng cho Hành chính/Luật)
                record_id = primary_task["procedure_id"]
                score = primary_task["procedure_score"]
                source_type = "SEMANTIC_PLANNER (Luật chính xác)"
            else:
                # [B] Fallback bằng Vector Search (Tự động áp dụng cho Tài chính, IT)
                print(f"--> [Planner] Phân tích luật không khớp (Status: {primary_task.get('status') if primary_task else 'None'}).")
                print(f"--> [Vector Search] Kích hoạt tìm kiếm theo ngữ nghĩa...")
                record_id, score, meta = retrieve_and_rerank(user_query, domain=domain)
                source_type = "VECTOR_SEARCH (Đoán ngữ nghĩa)"

            # 6. TRÍCH XUẤT VÀ LÀM SẠCH DỮ LIỆU TỪ SQLITE
            if record_id:
                full_row = await fetch_record_from_sqlite(str(record_id), domain=domain)
                if full_row:
                    # Lọc bỏ cột row_id hệ thống và lấp đầy các ô trống (Task 2)
                    filtered_data = {col: sanitize_value(val) for col, val in full_row.items() if col != "row_id"}
                    
                    # Lấy tên bài viết linh hoạt theo cấu trúc bảng
                    display_name = filtered_data.get('Tên thủ tục hành chính') or filtered_data.get('question') or 'N/A'
                    
                    print(f"\n[+] KẾT QUẢ TRUY XUẤT:")
                    print(f"    - Nguồn       : {source_type}")
                    print(f"    - ID Bản ghi  : {record_id} (Điểm tin cậy: {score:.4f})")
                    print(f"    - Tên bài viết: {display_name}")
                    print(f"    - Ý định hỏi  : {intent.upper()}")
                    print(f"    - Ngữ cảnh an toàn đã chuẩn bị cho LLM:")
                    print(json.dumps(filtered_data, ensure_ascii=False, indent=2))
                    
                    # Cập nhật lịch sử hội thoại cho Context Manager
                    ctx.update(user_query, resolved_tasks, [{"status": primary_task.get("status", "ready")}])
                else:
                    print("--> [Lỗi] Tìm thấy ID trên Qdrant nhưng SQLite không có dữ liệu tương ứng!")
            else:
                print("--> [Kết quả] Không tìm thấy dữ liệu phù hợp trong kho.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[!] Có lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    asyncio.run(main())