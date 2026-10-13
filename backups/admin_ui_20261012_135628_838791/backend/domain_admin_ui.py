"""Local Streamlit domain onboarding and configuration, alongside existing monitoring."""
import hashlib
import json
import os
from pathlib import Path
import re
import pandas as pd
import streamlit as st

from rag.domain_config import (ROOT_DIR, RUNTIME_KEYS, RAGConfigurationError, atomic_json,
    list_domain_configs, load_domain_config, mode_settings, read_json, validate_profile)
from rag.domain_onboarding import (propose_profile, publish_settings, generations,
    activate_generation, GENERIC_EVIDENCE_PROMPT)
from rag.ingest_pipeline import read_input, prepare_records
from rag.ingestion_jobs import start_job, list_jobs


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def ui_error(exc):
    st.error(str(exc))


def upload_page():
    st.subheader('Thêm dữ liệu / tạo lĩnh vực')
    st.caption('Tải file → Gemini đề xuất cấu hình → kiểm tra / chỉnh sửa → xác thực → ingest. '
               'Upload cho lĩnh vực đã có sẽ thay thế toàn bộ dữ liệu của lĩnh vực đó sau khi hoàn tất.')
    configs = list_domain_configs()
    existing = {c.name: c for c in configs}
    choice = st.selectbox('Lĩnh vực đích', ['Tạo lĩnh vực mới'] + list(existing), key='upload_domain')
    new = choice == 'Tạo lĩnh vực mới'
    domain_id = st.text_input('ID lĩnh vực mới (có thể để trống để Gemini đặt tên)',
                            placeholder='university / it_support / hr', key='new_domain_id') if new else choice
    file = st.file_uploader('Dataset dạng bảng', type=['xlsx', 'xlsm', 'csv', 'tsv', 'json', 'jsonl'], key='domain_upload')
    sheet_text = st.text_input('Sheet Excel: số thứ tự hoặc tên', value='0', key='upload_sheet')
    sheet = int(sheet_text) if sheet_text.isdecimal() else sheet_text
    if not file:
        st.info('Chọn file dữ liệu. Không cần viết Python hoặc tự tạo JSON trước.')
        return
    digest = hashlib.sha256(file.getvalue()).hexdigest()
    suffix = Path(file.name).suffix.lower()
    path = ROOT_DIR / 'data/admin_uploads' / (digest + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(file.getvalue())
    token = digest + '|' + choice + '|' + domain_id + '|' + str(sheet)
    if st.session_state.get('upload_token') != token:
        st.session_state['upload_token'] = token
        st.session_state.pop('profile_editor', None)
        st.session_state.pop('mapping_editor', None)
        if not new:
            st.session_state['profile_editor'] = dump(existing[choice].profile)
    try:
        df = read_input(path, sheet)
        st.write(f'{len(df)} dòng • {len(df.columns)} cột')
        st.dataframe(df.head(8), hide_index=True, use_container_width=True)
    except Exception as exc:
        ui_error(exc)
        return
    if new:
        st.caption('Gemini nhận tên cột và tối đa 8 dòng mẫu giới hạn độ dài; mô tả này phục vụ routing, '
                   'không thay thế dữ liệu gốc. API key lấy từ .env trên máy chủ.')
        if st.button('Gemini: tạo cấu hình tự động', type='primary'):
            try:
                with st.spinner('Gemini đang phân tích cấu trúc và phạm vi dữ liệu…'):
                    profile = propose_profile(df, domain_id.strip() or None)
                if profile['domain_name'] in existing:
                    raise RAGConfigurationError('ID Gemini đề xuất đã tồn tại. Nhập ID mới rồi tạo lại.')
                st.session_state['profile_editor'] = dump(profile)
                st.session_state.pop('mapping_editor', None)
                st.success('Đã tạo bản nháp. Kiểm tra cấu hình bên dưới rồi ingest.')
            except Exception as exc:
                ui_error(exc)
    else:
        st.info('Dùng lại cấu hình hiện tại; không gọi Gemini và không đổi schema tự động.')
    supplied = st.file_uploader('Hoặc nhập profile JSON đã có', type=['json'], key='import_profile')
    if supplied and st.button('Nạp profile JSON'):
        try:
            st.session_state['profile_editor'] = dump(validate_profile(json.loads(supplied.getvalue().decode('utf-8-sig'))))
            st.session_state.pop('mapping_editor', None)
        except Exception as exc:
            ui_error(exc)
    if 'profile_editor' not in st.session_state:
        return
    text = st.text_area('Profile đầy đủ (có thể chỉnh sửa)', height=300, key='profile_editor')
    try:
        profile = validate_profile(json.loads(text))
        if not new and profile['domain_name'] != choice:
            raise RAGConfigurationError('Không được đổi ID lĩnh vực hiện tại trong profile.')
        if new and profile['domain_name'] in existing:
            raise RAGConfigurationError('Lĩnh vực đã tồn tại. Chọn nó ở Lĩnh vực đích để cập nhật.')
        if new and domain_id.strip() and profile['domain_name'] != domain_id.strip():
            raise RAGConfigurationError('ID trong profile phải khớp ID lĩnh vực mới đã nhập.')
        st.markdown('**Ánh xạ cột và trường dùng để tìm kiếm**')
        mapping = pd.DataFrame([{'English key': k, 'Cột dữ liệu': v} for k, v in profile['schema_mapping'].items()])
        mapping_digest = hashlib.sha256(dump(profile['schema_mapping']).encode()).hexdigest()[:10]
        edited = st.data_editor(mapping, hide_index=True, num_rows='dynamic', use_container_width=True,
                               key='map_' + token[:12] + mapping_digest,
                               column_config={'Cột dữ liệu': st.column_config.SelectboxColumn(options=[str(c) for c in df.columns], required=True)})
        entries = [(str(row['English key']).strip(), str(row['Cột dữ liệu']).strip()) for _, row in edited.iterrows()]
        if len({k for k, _ in entries}) != len(entries):
            raise RAGConfigurationError('English keys phải duy nhất.')
        profile['schema_mapping'] = dict(entries)
        keys = list(profile['schema_mapping'])
        if not keys:
            raise RAGConfigurationError('Cần ít nhất một trường dữ liệu.')
        primary = profile['primary_entity']
        profile['primary_entity'] = st.selectbox('Trường tên / chủ đề chính', keys,
            index=keys.index(primary) if primary in keys else 0, key='primary_' + token[:12] + mapping_digest)
        profile['embedding_fields'] = st.multiselect('Các trường đưa vào embedding', keys,
            default=[k for k in profile.get('embedding_fields', [profile['primary_entity']]) if k in keys],
            key='embed_' + token[:12] + mapping_digest)
        profile['embedding_prefix'] = st.text_input('Tiền tố embedding', value=profile.get('embedding_prefix', ''),
                                                  key='prefix_' + token[:12])
        profile['excluded_fields'] = st.multiselect('Trường không hiển thị trong bằng chứng / Sources', keys,
            default=[k for k in profile.get('excluded_fields', ['timestamp', 'email', 'email_address']) if k in keys],
            key='excluded_' + token[:12] + mapping_digest)
        validate_profile(profile)
        _, report = prepare_records(df, profile)
        st.success(f"Hợp lệ: {report['indexed_rows']} bản ghi sẽ được ingest.")
        st.json(report)
        st.download_button('Tải profile đã kiểm tra', dump(profile), file_name=profile['domain_name'] + '_profile.json')
        if st.button('Bắt đầu ingest', type='primary'):
            job_id = start_job(path, profile, sheet=sheet)
            st.success(f'Đã tạo job {job_id}. Xem tab Tiến độ; dữ liệu cũ vẫn phục vụ cho đến khi job hoàn tất.')
    except Exception as exc:
        ui_error(exc)


def settings_page():
    st.subheader('Cấu hình theo lĩnh vực')
    configs = {c.name: c for c in list_domain_configs()}
    if not configs:
        st.info('Chưa có lĩnh vực. Upload và ingest trước.')
        return
    name = st.selectbox('Lĩnh vực cần cấu hình', list(configs), key='settings_domain')
    config = configs[name]
    st.caption('Thay đổi prompt / ngưỡng / số kết quả có hiệu lực ở lượt chat kế tiếp. '
               'Thay đổi schema hoặc embedding cần upload và ingest lại ở tab Thêm dữ liệu.')
    runtime = {k: v for k, v in config.profile.items() if k in RUNTIME_KEYS}
    if st.session_state.get('settings_identity') != name + config.revision:
        st.session_state['settings_identity'] = name + config.revision
        st.session_state['runtime_json'] = dump(runtime)
    form_key = name + config.revision[:10]
    baseline = {}
    prompt_path = ROOT_DIR / 'backend/prompts.json'
    if config.primary_entity == 'procedure_name' and prompt_path.exists():
        baseline = read_json(prompt_path).get('baseline_qa', {})
    with st.form('settings_' + form_key):
        description = st.text_area('Mô tả phạm vi dữ liệu (dùng phát hiện domain)', config.profile.get('description', ''), key='desc_' + form_key)
        examples = st.text_area('Câu hỏi minh họa cho routing (mỗi dòng một câu)', '\n'.join(config.profile.get('routing_examples', [])), key='examples_' + form_key)
        defaults = st.selectbox('Chế độ mặc định', ['strict', 'friendly'],
            index=0 if config.profile.get('default_mode', os.getenv('CHAT_MODE', 'strict')) == 'strict' else 1, key='mode_' + form_key)
        fallback = st.text_area('Thông báo khi không có bằng chứng (để trống giữ hành vi cũ)', config.profile.get('fallback_message', ''), key='fallback_' + form_key)
        exact = st.checkbox('Trả nguyên văn các trường được hỏi rõ', value=config.profile.get('exact_field_output', os.getenv('RAG_EXACT_FIELD_OUTPUT', '0') == '1' and config.primary_entity == 'procedure_name'), key='exact_' + form_key)
        st.caption('Khi trả nguyên văn trường dữ liệu, prompt và temperature không thay đổi các giá trị đó. '
                   'Các thiết lập generation áp dụng cho câu trả lời do LLM tạo.')
        modes = {}
        columns = st.columns(2)
        for col, mode in zip(columns, ['strict', 'friendly']):
            with col:
                st.markdown('**' + mode.capitalize() + '**')
                settings = {**baseline, **mode_settings(config, mode)}
                settings = {k: v for k, v in settings.items() if k in {'system_prompt', 'user_template', 'direct_prompt', 'temperature', 'top_p', 'max_tokens', 'minimum_score', 'candidate_k', 'result_k'}}
                settings['system_prompt'] = st.text_area('Prompt trả lời có dữ liệu', settings.get('system_prompt', GENERIC_EVIDENCE_PROMPT), height=180, key=mode + '_prompt_' + form_key)
                settings['user_template'] = st.text_area('Template: {context} và {query}', settings.get('user_template', 'RECORDS:\n{context}\nQUESTION:\n{query}'), key=mode + '_template_' + form_key)
                if mode == 'friendly':
                    settings['direct_prompt'] = st.text_area('Prompt giao tiếp không tra cứu', settings.get('direct_prompt', 'Bạn là trợ lý thân thiện. Trả lời ngắn gọn. Không tuyên bố đã tra cứu nguồn khi chưa tra cứu.'), key='direct_' + form_key)
                settings['temperature'] = st.number_input('Temperature', 0.0, 2.0, float(settings.get('temperature', 0.1 if mode == 'strict' else 0.3)), step=0.05, key=mode + '_temp_' + form_key)
                settings['top_p'] = st.number_input('Top-p', 0.01, 1.0, float(settings.get('top_p', 0.95)), step=0.01, key=mode + '_top_' + form_key)
                settings['max_tokens'] = st.number_input('Max output tokens', 1, 4096, int(settings.get('max_tokens', 800)), key=mode + '_tokens_' + form_key)
                settings['minimum_score'] = st.number_input('Ngưỡng RAG', 0.0, 1.0, float(settings.get('minimum_score', config.minimum_score)), step=0.01, format='%.4f', key=mode + '_score_' + form_key)
                settings['candidate_k'] = st.number_input('Số ứng viên', 1, 200, int(settings.get('candidate_k', config.retrieval_option('candidate_k', 'RAG_CANDIDATE_K', '20'))), key=mode + '_candidate_' + form_key)
                settings['result_k'] = st.number_input('Số bản ghi giữ lại', 1, 200, int(settings.get('result_k', config.retrieval_option('result_k', 'RAG_RESULT_K', '5'))), key=mode + '_result_' + form_key)
                modes[mode] = settings
        if st.form_submit_button('Áp dụng cấu hình', type='primary'):
            try:
                runtime.update(description=description, routing_examples=[x.strip() for x in examples.splitlines() if x.strip()],
                               default_mode=defaults, exact_field_output=exact, modes=modes)
                if fallback:
                    runtime['fallback_message'] = fallback
                else:
                    runtime.pop('fallback_message', None)
                publish_settings(name, runtime)
                st.success('Đã áp dụng. Không cần restart backend hoặc ingest lại.')
                st.rerun()
            except Exception as exc:
                ui_error(exc)
    with st.expander('Cấu hình nâng cao: aliases, routing, query preparation, selection, fallback'):
        raw = st.text_area('Runtime JSON (không chứa schema / đường dẫn storage)', key='runtime_json', height=350)
        if st.button('Áp dụng JSON nâng cao'):
            try:
                publish_settings(name, json.loads(raw))
                st.success('Đã áp dụng cấu hình nâng cao.')
                st.rerun()
            except Exception as exc:
                ui_error(exc)
        if st.button('Bỏ runtime overrides, dùng lại cấu hình snapshot'):
            path = ROOT_DIR / 'data/domain_settings' / f'{name}.json'
            path.unlink(missing_ok=True)
            st.rerun()


def status_page():
    st.subheader('Tiến độ ingest')
    if st.button('Làm mới trạng thái'):
        st.rerun()
    jobs = list_jobs()
    if not jobs:
        st.info('Chưa có job ingest từ admin UI.')
    for job in jobs[:20]:
        with st.expander(f"{job['domain']} • {job['status']} • {job['stage']}", expanded=job['status'] in {'running', 'queued', 'failed'}):
            st.write('Job:', job['id'])
            st.write('Cập nhật:', job['updated_at'])
            if job.get('error'):
                st.error(job['error'])
            st.json(job.get('report', {}))
            log_path = ROOT_DIR / 'data/admin_jobs' / (job['id'] + '.log')
            if log_path.exists():
                st.code(log_path.read_text(encoding='utf-8', errors='replace')[-12000:], language='text')
    st.subheader('Dữ liệu đang phục vụ và quay lại bản trước')
    configs = list_domain_configs()
    if not configs:
        return
    st.dataframe(pd.DataFrame([{'Domain': c.name, 'Generation': c.generation,
                              'Sẵn sàng': c.db_path.is_file() and c.vector_dir.is_dir(),
                              'Ngưỡng': c.minimum_score} for c in configs]), hide_index=True, use_container_width=True)
    name = st.selectbox('Domain cần xem phiên bản', [c.name for c in configs], key='rollback_domain')
    snapshots = generations(name)
    if not snapshots:
        st.caption('Domain này dùng dữ liệu legacy; phiên bản snapshot sẽ có sau lần ingest tiếp theo.')
        return
    generation = st.selectbox('Phiên bản', [g['generation'] for g in snapshots],
                             format_func=lambda x: x + (' (đang dùng)' if next(g for g in snapshots if g['generation'] == x)['active'] else ''))
    if st.button('Kích hoạt phiên bản đã chọn'):
        try:
            activate_generation(name, generation)
            st.success('Đã kích hoạt snapshot. Chat mới dùng đúng SQL và vectors của phiên bản đó.')
            st.rerun()
        except Exception as exc:
            ui_error(exc)


def render_domain_admin():
    st.header('Quản lý lĩnh vực và dữ liệu')
    upload, settings, status = st.tabs(['Thêm dữ liệu', 'Cấu hình', 'Tiến độ / phiên bản'])
    with upload:
        upload_page()
    with settings:
        settings_page()
    with status:
        status_page()
