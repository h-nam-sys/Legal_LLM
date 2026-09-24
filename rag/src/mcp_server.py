import sys
import json

from fastmcp import FastMCP


# ============================================================
# UTF-8
# ============================================================

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# FASTMCP SERVER
# ============================================================

mcp = FastMCP("Vietnamese Legal RAG")


# ============================================================
# RAG COMPONENTS
# ============================================================

model = None
client = None


def load_rag():
    global model, client

    if model is None or client is None:
        print(
            "Loading RAG components...",
            file=sys.stderr,
            flush=True
        )

        # Import ở đây để server MCP khởi động trước
        from retrieval import load_model, connect_qdrant

        model = load_model()
        client = connect_qdrant()

        print(
            "[OK] RAG components loaded",
            file=sys.stderr,
            flush=True
        )


# ============================================================
# MCP TOOL 1: SEARCH LAWS (LOCAL RAG)
# ============================================================

@mcp.tool
def search_laws(query: str) -> str:

    print(
        f"[MCP] Query: {query}",
        file=sys.stderr,
        flush=True
    )

    # Load BKAI + Qdrant khi tool được gọi
    load_rag()

    from retrieval import retrieve

    results = retrieve(
        query=query,
        model=model,
        client=client
    )

    output = []

    for rank, item in enumerate(results, start=1):

        result = item["result"]
        payload = result.payload or {}

        output.append({
            "rank": rank,

            "score": round(
                float(item["final_score"]),
                4
            ),

            "chunk_id": payload.get(
                "chunk_id",
                ""
            ),

            "document_id": payload.get(
                "document_id",
                ""
            ),

            "procedure_name": payload.get(
                "procedure_name",
                ""
            ),

            "field": payload.get(
                "field",
                ""
            ),

            "submission_method": payload.get(
                "submission_method",
                ""
            ),

            "chunk_type": payload.get(
                "chunk_type",
                ""
            ),

            "text": payload.get(
                "text",
                ""
            )
        })

    return json.dumps(
        output,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# MCP TOOL 2: SEARCH ONLINE (DICHVUCONG ONLY)
# ============================================================

@mcp.tool
def search_online_dichvucong(query: str, location: str = "") -> str:
    """
    Tìm kiếm trực tuyến trên Cổng Dịch vụ công Quốc gia.
    Chỉ gọi tool này khi RAG Local không có kết quả VÀ người dùng đã đồng ý tìm mạng.
    """

    # 1. Khóa cứng domain để chống nhiễu
    domain_restriction = "site:dichvucong.gov.vn"

    # 2. Ép thêm vị trí địa lý (Geofencing)
    geo_context = f"tại {location}" if location else ""

    # 3. Nối thành câu truy vấn chuẩn (Strict Query)
    strict_query = f"{query} {geo_context} {domain_restriction}".strip()

    print(
        f"[MCP] Executing Online Search with strict query: {strict_query}",
        file=sys.stderr,
        flush=True
    )

    # ---------------------------------------------------------
    # CHỖ NÀY DÙNG ĐỂ GỌI API SEARCH (Tavily, Google Custom Search, v.v)
    # Ví dụ:
    # results = call_my_search_api(strict_query)
    # return json.dumps(results, ensure_ascii=False)
    # ---------------------------------------------------------

    # Mock response tạm thời trước khi bạn đấu nối API thật
    mock_response = {
        "status": "pending_implementation",
        "executed_query": strict_query,
        "message": "Truy vấn đã được khóa domain. Hãy thay thế block này bằng API tìm kiếm web thực tế của bạn."
    }

    return json.dumps(
        mock_response,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# MCP TOOL 3: SYNC LAWS
# ============================================================

@mcp.tool
def sync_laws() -> str:

    print(
        "[MCP] Starting realtime legal data sync...",
        file=sys.stderr,
        flush=True
    )

    try:
        # Import realtime sync khi tool được gọi
        from realtime_sync import sync

        result = sync()

        print(
            "[MCP] Realtime sync completed",
            file=sys.stderr,
            flush=True
        )

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )

    except Exception as e:

        print(
            f"[MCP] Sync failed: {e}",
            file=sys.stderr,
            flush=True
        )

        return json.dumps(
            {
                "status": "error",
                "message": str(e)
            },
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    mcp.run()
