"""Simple MCP administration with advanced connection settings collapsed."""
import asyncio
import html
import copy
import json
import os
import streamlit as st
from rag.domain_config import list_domain_configs, load_domain_config
from rag.mcp_config import load_settings, save_settings, validate_settings, rule_for, MCPConfigError
from backend.services.mcp_service import inspect_tools, retrieve_external, MCPSearchError


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

@timed('backend.mcp_admin_ui.render_mcp_admin')
def render_mcp_admin():
    st.subheader('Tìm kiếm trực tuyến (MCP)')
    st.caption('Chỉ tra cứu khi dữ liệu nội bộ không đủ và người dùng chọn Có. Cấu hình được áp dụng cho lượt tra cứu tiếp theo.')
    try:
        current = load_settings()
        configs = list_domain_configs()
    except Exception as exc:
        st.error(str(exc))
        return
    draft = copy.deepcopy(current)
    draft['enabled'] = st.toggle('Cho phép tìm kiếm trực tuyến', value=current['enabled'], key='mcp_enabled')
    by_name = {c.name: c for c in configs}
    selected = st.multiselect('Lĩnh vực được phép tìm kiếm', options=list(by_name),
        default=[c.name for c in configs if rule_for(c, current)['enabled']], key='mcp_domains')
    # Preserve disabled domain rules; rendering this page never writes configuration.
    for name, config in by_name.items():
        draft['domain_rules'][name] = {**rule_for(config, current), 'enabled': name in selected}
    if selected:
        domain = st.selectbox('Website tìm kiếm của lĩnh vực', selected, key='mcp_rule_domain')
        rule = draft['domain_rules'][domain]
        sites = st.text_area('Chỉ tìm trên các website này (mỗi dòng một tên miền)',
            value='\n'.join(rule['allowed_hosts']), key='mcp_sites_' + domain,
            placeholder='dichvucong.gov.vn')
        rule['allowed_hosts'] = [s.strip() for s in sites.splitlines() if s.strip()]
        st.caption('Để trống để tìm trên web công khai. Ví dụ lĩnh vực trường học: website chính thức của trường.')
        rule['use_location'] = st.checkbox('Thêm địa điểm của cuộc trò chuyện vào câu hỏi tìm kiếm',
            value=rule['use_location'], key='mcp_location_' + domain)
    # Retain edited website/location rules when the admin switches the domain selector.
    for name in by_name:
        if 'mcp_sites_' + name in st.session_state:
            draft['domain_rules'][name]['allowed_hosts'] = [x.strip() for x in st.session_state['mcp_sites_' + name].splitlines() if x.strip()]
        if 'mcp_location_' + name in st.session_state:
            draft['domain_rules'][name]['use_location'] = st.session_state['mcp_location_' + name]
    st.info('Máy chủ tích hợp dùng Gemini với Google Search, dùng lại GEMINI_API_KEY hiện có. Qwen tổng hợp câu trả lời từ các thông tin được Gemini dẫn nguồn.')
    st.caption('Khóa Gemini trên máy chủ: ' + ('Đã cấu hình' if os.getenv('GEMINI_API_KEY') else 'Chưa cấu hình'))
    with st.expander('Cấu hình nâng cao', expanded=False):
        transport = st.selectbox('Kết nối MCP', ['bundled', 'http'],
            index=['bundled', 'http'].index(current['transport']),
            format_func=lambda x: 'Máy chủ tìm kiếm tích hợp' if x == 'bundled' else 'Máy chủ MCP từ xa (Streamable HTTP)', key='mcp_transport')
        draft['transport'] = transport
        if transport == 'http':
            draft['server_url'] = st.text_input('URL máy chủ MCP', value=current['server_url'], key='mcp_url')
            draft['token_env'] = st.text_input('Tên biến môi trường chứa Bearer token (nếu cần)', value=current['token_env'], key='mcp_token_env')
            draft['tool_name'] = st.text_input('Tên công cụ tìm kiếm', value=current['tool_name'], key='mcp_tool')
            draft['query_argument'] = st.text_input('Tên tham số câu hỏi', value=current['query_argument'], key='mcp_query_arg')
            draft['count_argument'] = st.text_input('Tên tham số số kết quả (có thể để trống)', value=current['count_argument'], key='mcp_count_arg')
            extras = st.text_area('Tham số bổ sung (JSON, không chứa khóa API)', value=json.dumps(current['extra_arguments'], ensure_ascii=False, indent=2), key='mcp_extra')
            try:
                draft['extra_arguments'] = json.loads(extras)
            except ValueError:
                draft['extra_arguments'] = None
            st.caption('Công cụ cần trả JSON results: danh sách các đối tượng title, url, snippet. Máy chủ phải hỗ trợ các tham số đã khai báo.')
        else:
            draft.update(tool_name='web_search', query_argument='query', count_argument='count', extra_arguments={})
        if transport == 'bundled':
            draft['gemini_model'] = st.text_input('Model Gemini dùng Google Search', value=current['gemini_model'], key='mcp_gemini_model')
        draft['result_limit'] = st.number_input('Số nguồn tối đa', min_value=1, max_value=10, value=current['result_limit'], key='mcp_limit')
        draft['timeout_seconds'] = st.number_input('Thời gian chờ MCP (giây)', min_value=5, max_value=180, value=current['timeout_seconds'], key='mcp_timeout')
        draft['max_tokens'] = st.number_input('Độ dài trả lời tối đa (token)', min_value=100, max_value=2000, value=current['max_tokens'], key='mcp_tokens')
        draft['answer_prompt'] = st.text_area('Hướng dẫn trả lời từ nguồn ngoài', value=current['answer_prompt'], height=160, key='mcp_prompt')
    c1, c2 = st.columns(2)
    if c1.button('Lưu cấu hình MCP', type='primary'):
        try:
            save_settings(draft)
            st.success('Đã lưu cấu hình MCP. Không cần nhập lại dữ liệu.')
        except Exception as exc:
            st.error(str(exc))
    if c2.button('Kiểm tra kết nối'):
        try:
            clean = validate_settings(draft)
            with st.spinner('Đang kết nối máy chủ MCP…'):
                tools = asyncio.run(inspect_tools(clean))
            names = [t['name'] for t in tools]
            if clean['tool_name'] not in names:
                st.warning('Kết nối được, nhưng chưa có công cụ đã chọn: ' + clean['tool_name'])
            else:
                st.success('Kết nối MCP thành công. Công cụ đã chọn có sẵn.')
            st.write('Công cụ có sẵn: ' + ', '.join(names))
            st.caption('Kiểm tra kết nối không gọi API tìm kiếm và không kiểm tra khóa Gemini.')
            with st.expander('Chi tiết công cụ'):
                st.json(tools)
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Thử tra cứu', expanded=False):
        st.caption('Thử bằng cấu hình đang hiển thị, không lưu bản nháp. Nút này thực hiện một lượt tra cứu qua nhà cung cấp tìm kiếm.')
        test_domain = st.selectbox('Lĩnh vực thử', selected, key='mcp_test_domain') if selected else None
        question = st.text_input('Câu hỏi thử', key='mcp_test_query')
        if st.button('Tra cứu thử', disabled=not test_domain or not question.strip()):
            try:
                clean = validate_settings(draft)
                # An explicit admin test is authorized even before the global toggle is published.
                clean['enabled'] = True
                with st.spinner('Đang tra cứu qua MCP…'):
                    records = asyncio.run(retrieve_external(question, load_domain_config(test_domain), settings=clean))
                if not records:
                    st.warning('Không có đoạn trích phù hợp từ các website được phép.')
                if getattr(records, 'search_suggestions', ''):
                    doc = html.escape(records.search_suggestions, quote=True)
                    st.markdown(f'<iframe title="Google Search" sandbox="allow-popups allow-popups-to-escape-sandbox" srcdoc="{doc}" style="width:100%;height:150px;border:0"></iframe>', unsafe_allow_html=True)
                for r in records:
                    st.write(r['title'])
                    st.code(r['url'], language=None)
                    if r.get('evidence_kind') == 'gemini_grounded_summary':
                        st.caption('Tóm tắt Gemini có dẫn nguồn (không phải trích nguyên văn website)')
                    st.text(r['snippet'])
            except Exception as exc:
                st.error(str(exc))
