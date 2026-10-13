import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import requests

# --- CONFIG ---
DB_PATH = "./legal_llm.db"
FEEDBACK_FILE = "./all_feedback_dataset.jsonl"
API_URL = "http://localhost:8000" # Update this if your FastAPI runs on a different port/host

st.set_page_config(page_title="Admin Dashboard | Tăng Nhơn Phú", layout="wide")

# --- DATA FETCHING ---
@st.cache_data(ttl=60) # Cache for 60 seconds to prevent DB locking
def load_db_data(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_feedback_data():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    data = []
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                pass
    return pd.DataFrame(data)

# --- UI LAYOUT ---
st.title("🏛️ Bảng Điều Khiển Trợ Lý Pháp Lý")
st.markdown("Giám sát hoạt động, chất lượng phản hồi và dữ liệu người dùng tại phường Tăng Nhơn Phú.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan (Overview)", "🕵️ Nhật ký hoạt động (Audit Logs)", "⭐ Dữ liệu Đánh giá (Feedback)", "⚠️ Truy vấn Thất bại (Failed Queries)"])

# ==========================================
# TAB 1: OVERVIEW & STATS
# ==========================================
with tab1:
    st.subheader("Chỉ số hệ thống (Real-time)")

    # KPIs
    conv_df = load_db_data("SELECT id, created_at FROM conversations")
    msg_df = load_db_data("SELECT id, role, created_at FROM messages")
    audit_df = load_db_data("SELECT id, used_online_search FROM audit_logs")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Phiên Tư Vấn", len(conv_df))
    col2.metric("Tổng Tin Nhắn", len(msg_df))

    mcp_count = len(audit_df[audit_df['used_online_search'] == 1]) if not audit_df.empty else 0
    col3.metric("Lượt gọi MCP Web Search", mcp_count)

    user_msgs = len(msg_df[msg_df['role'] == 'user']) if not msg_df.empty else 0
    col4.metric("Truy vấn từ người dân", user_msgs)

    st.divider()

    # Chart: Messages over time
    if not msg_df.empty:
        st.subheader("Lưu lượng truy cập theo ngày")
        msg_df['created_at'] = pd.to_datetime(msg_df['created_at'])
        daily_counts = msg_df.groupby(msg_df['created_at'].dt.date).size()
        st.bar_chart(daily_counts)

# ==========================================
# TAB 2: AUDIT LOGS (MONITORING)
# ==========================================
with tab2:
    st.subheader("Nhật ký tra cứu & Trả lời (Audit Logs)")
    st.markdown("Theo dõi chính xác LLM đã trả lời gì và dùng ngữ cảnh nào.")

    # UPDATED QUERY TO INCLUDE NEW COLUMNS IF THEY EXIST
    try:
        logs = load_db_data("SELECT created_at, conversation_id, user_query, llm_final_response, used_online_search, retrieved_local_context, rag_score, failure_type FROM audit_logs ORDER BY created_at DESC LIMIT 100")
    except:
         logs = load_db_data("SELECT created_at, conversation_id, user_query, llm_final_response, used_online_search, retrieved_local_context FROM audit_logs ORDER BY created_at DESC LIMIT 100")


    if not logs.empty:
        # Style the dataframe
        logs['used_online_search'] = logs['used_online_search'].apply(lambda x: "🌐 Web Search" if x == 1 else "📁 Qdrant Local")

        column_cfg = {
                "created_at": "Thời gian",
                "conversation_id": "ID Phiên",
                "user_query": "Người dân hỏi",
                "llm_final_response": "AI Trả lời",
                "used_online_search": "Nguồn RAG",
                "retrieved_local_context": "Văn bản trích xuất"
            }

        if 'rag_score' in logs.columns:
            column_cfg["rag_score"] = "Điểm RAG"
        if 'failure_type' in logs.columns:
             column_cfg["failure_type"] = "Trạng thái"

        st.dataframe(
            logs,
            column_config=column_cfg,
            hide_index=True,
            use_container_width=True,
            height=500
        )
    else:
        st.info("Chưa có nhật ký hoạt động nào trong Database.")

# ==========================================
# TAB 3: FEEDBACK (RLHF / FINE-TUNING)
# ==========================================
with tab3:
    st.subheader("Dữ liệu đánh giá từ người dân (SFT / RLHF)")
    st.markdown("Dữ liệu này được lưu trực tiếp vào JSONL để sẵn sàng Fine-tune mô hình.")

    feedback_df = load_feedback_data()

    if not feedback_df.empty:
        # Convert 1/0 to emoji
        feedback_df['rating_display'] = feedback_df['rating'].apply(lambda x: "👍 Tốt" if x == 1 else "👎 Kém")

        # Stats
        good_count = len(feedback_df[feedback_df['rating'] == 1])
        bad_count = len(feedback_df[feedback_df['rating'] == 0])

        col1, col2 = st.columns(2)
        col1.metric("👍 Số lượt đánh giá Tốt", good_count)
        col2.metric("👎 Số lượt đánh giá Kém", bad_count)

        st.dataframe(
            feedback_df[['timestamp', 'rating_display', 'input', 'output', 'rag_context']],
            column_config={
                "timestamp": "Thời gian",
                "rating_display": "Đánh giá",
                "input": "Câu hỏi",
                "output": "Câu trả lời của AI",
                "rag_context": "Ngữ cảnh cung cấp"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Chưa có đánh giá nào từ người dùng.")

# ==========================================
# TAB 4: FAILED QUERIES (NEW)
# ==========================================
with tab4:
    st.subheader("⚠️ Truy vấn Thất bại (Teencode / Thiếu dữ liệu)")
    st.markdown("Danh sách các câu hỏi khiến RAG trả về điểm thấp hoặc phải chuyển qua tìm kiếm MCP. Sử dụng dữ liệu này để bổ sung từ khóa hoặc cập nhật CSDL.")

    if st.button("🔄 Tải lại dữ liệu"):
        st.cache_data.clear()

    try:
        response = requests.get(f"{API_URL}/admin/failed-logs")
        if response.status_code == 200:
            failed_logs = response.json()
            if failed_logs:
                df_failed = pd.DataFrame(failed_logs)

                st.dataframe(
                    df_failed[["created_at", "failure_type", "rag_score", "user_query", "llm_response"]],
                    column_config={
                        "created_at": "Thời gian",
                        "failure_type": "Loại lỗi",
                        "rag_score": "Điểm RAG",
                        "user_query": "Câu hỏi thô (Raw Query)",
                        "llm_response": "AI Phản hồi"
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500
                )
            else:
                st.success("Tuyệt vời! Hiện tại không ghi nhận truy vấn thất bại nào.")
        else:
            st.error(f"Lỗi API: {response.status_code}. Vui lòng kiểm tra xem Backend FastAPI đã chạy chưa.")
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến Backend FastAPI. Đảm bảo server đang chạy tại localhost:8000.")
