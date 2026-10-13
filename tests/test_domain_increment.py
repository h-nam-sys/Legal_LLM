import asyncio
from dataclasses import replace
import importlib
import os
import json
from pathlib import Path
import sqlite3
import sys

import httpx
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Tests never use uploaded credentials or environment proxy settings.
os.environ['GEMINI_API_KEY'] = ''
for key in ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'backend'))
from rag import domain_config as dc
from rag import domain_onboarding as onboarding
from rag import ingest_pipeline as ingest
from rag import ingestion_jobs as jobs
from rag import query_pipeline as qp
from rag.query_preparation import PreparedQueries
from services import domain_router as routing
from services import llm_service as answering
from services import chat_orchestrator as chat
from services.answer_policy import resolve_mode
from schemas import ChatRequest
from database import Base
from services import db_service


def profile(name, legal=False):
    if legal:
        return {'domain_name': name, 'primary_entity': 'procedure_name',
                'schema_mapping': {'procedure_name': 'Tên thủ tục', 'fee': 'Lệ phí', 'processing_time': 'Thời gian'},
                'minimum_score': 0.01}
    return {'domain_name': name, 'primary_entity': 'topic',
            'schema_mapping': {'topic': 'Topic', 'details': 'Details'}, 'minimum_score': 0.01,
            'description': 'University student rules', 'field_aliases': {'details': ['chi tiết', 'details']},
            'exact_field_output': True, 'allow_forms': False, 'allow_external_search': False,
            'default_mode': 'strict'}


def fake_vectors(records, profile, directory):
    directory.mkdir()
    (directory / 'docstore.json').write_text(json.dumps([r['id'] for r in records]))
    (directory / 'index_store.json').write_text('{}')


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for module in [dc, onboarding, ingest, jobs, answering, routing]:
        monkeypatch.setattr(module, 'ROOT_DIR', tmp_path)
    # The baseline prompt path is needed only for legacy administrative generation.
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'backend/prompts.json').write_text((PROJECT / 'backend/prompts.json').read_text())
    monkeypatch.setenv('RAG_PREPARE_QUERY', '1')
    monkeypatch.setenv('RAG_EXACT_FIELD_OUTPUT', '1')
    monkeypatch.delenv('RAG_MIN_SCORE', raising=False)
    monkeypatch.delenv('CHAT_MODE', raising=False)
    answering.qa_cache.clear()
    return tmp_path


def publish(root, p, rows):
    df = pd.DataFrame(rows)
    records, report = ingest.prepare_records(df, p)
    config = onboarding.config_from_profile(p, root=root)
    generation = ingest.publish_generation(config, records, report, vector_builder=fake_vectors)
    onboarding.save_profile_registry(p, root=root)
    return dc.load_domain_config(p['domain_name'], root=root), generation


def test_sql_vector_id_pair_and_domain_isolation(roots):
    legal, g1 = publish(roots, profile('legal', True), [{'Tên thủ tục': 'Thủ tục đăng ký kết hôn', 'Lệ phí': '0; bản sao 8.000đ', 'Thời gian': 'Trong ngày'}])
    university, g2 = publish(roots, profile('university'), [{'Topic': 'Scholarship', 'Details': 'GPA 3.5'}])
    assert legal.db_path != university.db_path
    assert legal.vector_dir != university.vector_dir
    for c in [legal, university]:
        with sqlite3.connect(c.db_path) as conn:
            rows = conn.execute('SELECT id, _raw_json FROM records').fetchall()
        assert json.loads((c.vector_dir / 'docstore.json').read_text()) == [row[0] for row in rows]
        assert json.loads(rows[0][1])
    assert dc.load_domain_config(root=roots).name == 'legal'
    assert {c.name for c in dc.list_domain_configs(root=roots)} == {'legal', 'university'}


def test_vector_failure_does_not_activate_partial_snapshot(roots):
    c, old = publish(roots, profile('legal', True), [{'Tên thủ tục': 'Kết hôn', 'Lệ phí': '0', 'Thời gian': '1 ngày'}])
    records, report = ingest.prepare_records(pd.DataFrame([{'Tên thủ tục': 'Khai sinh', 'Lệ phí': '0', 'Thời gian': '1 ngày'}]), c.profile)
    def fail(*args):
        raise RuntimeError('Simulated vector build failure')
    with pytest.raises(RuntimeError):
        ingest.publish_generation(c, records, report, vector_builder=fail)
    assert dc.load_domain_config('legal', root=roots).generation == old
    assert len(onboarding.generations('legal', root=roots)) == 1


def test_validation_prevents_lossy_or_invalid_configuration(roots):
    p = profile('university')
    for update in [{'minimum_score': float('nan')}, {'candidate_k': 2, 'result_k': 5},
                   {'schema_mapping': {'bad;sql': 'Topic'}},
                   {'modes': {'strict': {'user_template': '{context}{query}{other}'}}},
                   {'field_aliases': {'missing': ['details']}}]:
        with pytest.raises((dc.RAGConfigurationError, ValueError)):
            dc.validate_profile({**p, **update})
    with pytest.raises(dc.RAGConfigurationError):
        ingest.prepare_records(pd.DataFrame([{'Wrong': 'x'}]), p)
    with pytest.raises(dc.RAGConfigurationError):
        ingest.prepare_records(pd.DataFrame([{'Topic': '' , 'Details': 'x'}]), p)


def test_source_ids_and_reordering_are_stable():
    p = profile('u')
    df = pd.DataFrame([{'Topic': 'A', 'Details': '1'}, {'Topic': 'B', 'Details': '2'}])
    a, _ = ingest.prepare_records(df, p)
    b, _ = ingest.prepare_records(df.iloc[::-1], p)
    assert {r['topic']: r['id'] for r in a} == {r['topic']: r['id'] for r in b}


@pytest.mark.parametrize('suffix', ['csv', 'tsv', 'xlsx', 'json', 'jsonl'])
def test_supported_upload_formats(roots, suffix):
    df = pd.DataFrame([{'Topic': 'A', 'Details': 'NA'}])
    path = roots / ('input.' + suffix)
    if suffix == 'xlsx':
        df.to_excel(path, index=False)
    elif suffix in {'csv', 'tsv'}:
        df.to_csv(path, index=False, sep='\t' if suffix == 'tsv' else ',')
    elif suffix == 'json':
        path.write_text(df.to_json(orient='records'))
    else:
        path.write_text(df.to_json(orient='records', lines=True))
    out = ingest.read_input(path)
    assert out.iloc[0]['Details'] == 'NA'
    ingest.prepare_records(out, profile('u'))


def test_gemini_proposal_uses_bounded_sample_and_exact_mapping(roots, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-test-key')
    generated = {k: v for k, v in profile('university').items()
                 if k in {'domain_name', 'primary_entity', 'schema_mapping', 'description', 'field_aliases'}}
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers['x-goog-api-key'] == 'fake-test-key'
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': json.dumps(generated)}]}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        p = onboarding.propose_profile(pd.DataFrame([{'Topic': str(i), 'Details': 'long text' * 1000} for i in range(100)]), 'campus', client=client)
    sample = json.loads(seen[0]['contents'][0]['parts'][0]['text'])
    assert len(sample['sample_rows']) == 8
    assert len(json.dumps(sample)) < 30000
    assert p['domain_name'] == 'campus'
    assert p['schema_mapping']['details'] == 'Details'
    assert p['embedding_fields'] == ['topic']
    assert not (roots / 'data/domain_profile.json').exists()


def test_invalid_gemini_response_has_no_silent_legal_fallback(roots, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-test-key')
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': '{}'}]}}]}))) as client:
        with pytest.raises(dc.RAGConfigurationError):
            onboarding.propose_profile(pd.DataFrame([{'Topic': 'x', 'Details': 'y'}]), client=client)
    assert not (roots / 'data').exists()


def test_runtime_changes_apply_without_reingestion_and_invalidate_cache(roots):
    c, g = publish(roots, profile('u'), [{'Topic': 'A', 'Details': 'B'}])
    updated = onboarding.publish_settings('u', {'description': 'Updated', 'modes': {
        'strict': {'minimum_score': 0.7, 'candidate_k': 10, 'result_k': 2, 'temperature': 0.0},
        'friendly': {'minimum_score': 0.2, 'candidate_k': 15, 'result_k': 5, 'temperature': 0.4}}}, root=roots)
    assert updated.generation == g
    assert updated.revision != c.revision
    assert dc.config_for_mode(updated, 'strict').minimum_score == 0.7
    assert dc.config_for_mode(updated, 'friendly').minimum_score == 0.2
    with pytest.raises(dc.RAGConfigurationError):
        onboarding.publish_settings('u', {'schema_mapping': {'x': 'X'}}, root=roots)
    assert dc.load_domain_config('u', root=roots).revision == updated.revision


def test_rollback_and_writer_lock(roots):
    p = profile('u')
    _, first = publish(roots, p, [{'Topic': 'A', 'Details': 'Old'}])
    _, second = publish(roots, p, [{'Topic': 'A', 'Details': 'New'}])
    with jobs.domain_lock('u', root=roots):
        with pytest.raises(dc.RAGConfigurationError):
            onboarding.activate_generation('u', first, root=roots)
    onboarding.activate_generation('u', first, root=roots)
    assert dc.load_domain_config('u', root=roots).generation == first
    assert first != second
    with pytest.raises(dc.RAGConfigurationError):
        onboarding.activate_generation('u', '../other', root=roots)


def test_background_job_success_and_failure_report(roots, monkeypatch):
    p = profile('u')
    path = roots / 'upload.csv'
    pd.DataFrame([{'Topic': 'A', 'Details': 'B'}]).to_csv(path, index=False)
    # Don't launch an actual model subprocess; exercise the real durable worker with fake vectors.
    class Process:
        pid = 999999999
    monkeypatch.setattr(jobs.subprocess, 'Popen', lambda *a, **k: Process())
    job_id = jobs.start_job(path, p, root=roots)
    job_path = roots / 'data/admin_jobs' / (job_id + '.json')
    assert jobs.run_job(job_path, vector_builder=fake_vectors)
    assert dc.read_json(job_path)['status'] == 'succeeded'
    before = dc.load_domain_config('u', root=roots).generation
    job_id = jobs.start_job(path, p, root=roots)
    def fail(*args):
        raise RuntimeError('build failed')
    assert not jobs.run_job(roots / 'data/admin_jobs' / (job_id + '.json'), vector_builder=fail)
    assert dc.load_domain_config('u', root=roots).generation == before
    assert jobs.list_jobs(root=roots)[0]['status'] == 'failed'
    assert not jobs.lock_path('u', roots).exists()


def two_domains(roots):
    publish(roots, profile('legal', True), [{'Tên thủ tục': 'Thủ tục đăng ký kết hôn', 'Lệ phí': 'Bản chính 0; bản sao 8.000đ', 'Thời gian': 'Trong ngày'}])
    publish(roots, profile('university'), [{'Topic': 'Scholarship', 'Details': 'GPA 3.5'}])


def fake_async_client(monkeypatch, decision):
    def handler(request):
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(decision)}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(routing.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_explicit_and_property_followup_skip_domain_llm(roots, monkeypatch):
    two_domains(roots)
    def fail(**kwargs):
        raise AssertionError('Should not call routing LLM')
    monkeypatch.setattr(routing.httpx, 'AsyncClient', fail)
    c, state, changed = asyncio.run(routing.choose_domain('Scholarship details', {}, [], 'university'))
    assert c.name == 'university' and state['selection'] == 'manual'
    c, _, _ = asyncio.run(routing.choose_domain('kết hôn', state, ['Scholarship'], None))
    assert c.name == 'university'
    c, _, changed = asyncio.run(routing.choose_domain('lệ phí bao nhiêu?', {'name': 'legal', 'selection': 'auto'}, ['Thủ tục đăng ký kết hôn'], 'auto'))
    assert c.name == 'legal' and not changed


def test_auto_topic_switch_and_invalid_decision_clarify(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'university'})
    c, state, changed = asyncio.run(routing.choose_domain('Scholarship requirements', {'name': 'legal', 'selection': 'auto'}, ['Marriage']))
    assert c.name == 'university' and changed
    with pytest.raises(ValueError):
        routing.parse_domain_decision('{"kind":"domain","domain":"unknown"}', {'legal'})


def test_legacy_modes_and_exact_rendering_are_preserved(roots, monkeypatch):
    config = onboarding.config_from_profile(profile('legal', True), root=roots)
    monkeypatch.setenv('CHAT_MODE', 'friendly')
    assert resolve_mode(None, config) == 'friendly'
    assert resolve_mode('strict', config) == 'strict'
    context = json.dumps({'Tên thủ tục': 'Kết hôn', 'Lệ phí': 'Bản chính: Không thu phí; Bản sao: 8.000đ/bản'}, ensure_ascii=False)
    rendered = answering.render_explicit_lookup(context, 'mất bao nhiêu tiền?', config)
    assert '8.000đ/bản' in rendered and 'Không thu phí' in rendered
    assert answering.render_explicit_lookup(context, 'ở Hà Nội lệ phí bao nhiêu?', config) is None


def test_chat_domain_isolation_followup_and_full_sources(roots, monkeypatch):
    two_domains(roots)
    def retrieve(phrase, *, config):
        with sqlite3.connect(config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute('SELECT * FROM records')]
        return [(row['id'], 0.99, {config.primary_entity: row[config.primary_entity]}) for row in rows]
    async def prep(prompt, search, config):
        return PreparedQueries(('Thủ tục đăng ký kết hôn' if config.name == 'legal' else 'Scholarship',), 'prepared')
    async def fields(*args, **kwargs):
        return '[]'
    monkeypatch.setattr(qp, 'retrieve_and_rerank', retrieve)
    monkeypatch.setattr(qp, 'prepare_search_queries', prep)
    monkeypatch.setattr(qp, 'call_llm_json', fields)
    async def run():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'chat.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as db:
            conv = await db_service.create_conversation(db, 'user1')
            first = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='đăng ký kết hôn mất bao nhiêu tiền?', domain='legal'), db)
            text = ''.join([part async for part in first.body_iterator])
            assert '8.000đ' in text and 'RAG_DOMAIN:legal' in text
            second = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='Scholarship details', domain='university'), db)
            text = ''.join([part async for part in second.body_iterator])
            assert 'GPA 3.5' in text and '8.000đ' not in text
            conv = await db_service.get_conversation(db, conv.id)
            assert conv.checklist['_domain_state']['name'] == 'university'
            history = await db_service.get_conversation_history(db, conv.id)
            assert chat.active_procedures(history, 'Marriage', 'university') == ['Scholarship']
            assert chat.active_procedures(history, 'Marriage', 'legal') == []
            third = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='details?', domain='auto'), db)
            text = ''.join([part async for part in third.body_iterator])
            assert 'GPA 3.5' in text and text.count('data: [DONE]') == 1
            history = await db_service.get_conversation_history(db, conv.id)
            last = history[-1]
            sources = json.loads(last.sources)
            assert any(x.startswith('RAG_CONTENT:') for x in sources)
            assert 'RAG_DOMAIN:university' in sources
        await engine.dispose()
    asyncio.run(run())


def test_ambiguous_chat_returns_reviewable_clarification(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'ambiguous', 'domain': None})
    async def run():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'chat.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            conv = await db_service.create_conversation(db, 'user1')
            response = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='quy định đăng ký?', domain='auto'), db)
            text = ''.join([chunk async for chunk in response.body_iterator])
            assert 'domain_choices' in text and 'legal' in text and 'university' in text
            assert text.count('data: [DONE]') == 1
        await engine.dispose()
    asyncio.run(run())


def test_fastapi_contracts_domains_history_and_streaming(roots, monkeypatch):
    two_domains(roots)
    monkeypatch.chdir(PROJECT / 'backend')
    main = importlib.import_module('main')
    async def retrieve_fields(*args, **kwargs):
        return '[]'
    async def prep(prompt, search, config):
        return PreparedQueries(('Scholarship',), 'prepared')
    def retrieve(phrase, *, config):
        with sqlite3.connect(config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute('SELECT * FROM records').fetchone())
        return [(row['id'], 0.99, {config.primary_entity: row[config.primary_entity]})]
    monkeypatch.setattr(qp, 'call_llm_json', retrieve_fields)
    monkeypatch.setattr(qp, 'prepare_search_queries', prep)
    monkeypatch.setattr(qp, 'retrieve_and_rerank', retrieve)
    async def run():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'api.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async def dependency():
            async with sessions() as db:
                yield db
        from database import get_db
        main.app.dependency_overrides[get_db] = dependency
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
                response = await client.get('/domains')
                assert response.status_code == 200
                assert {d['id'] for d in response.json()} == {'legal', 'university'}
                login = (await client.post('/mock-login', json={'username': 'test'})).json()
                headers = {'X-User-ID': login['user_id']}
                response = await client.post('/start-conversation', json={'title': 'Test', 'domain': 'university'}, headers=headers)
                assert response.status_code == 200
                cid = response.json()['conversation_id']
                response = await client.post(f'/chat/{cid}', json={'user_prompt': 'Scholarship details', 'domain': 'university'}, headers=headers)
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('text/event-stream')
                assert 'GPA 3.5' in response.text and response.text.count('data: [DONE]') == 1
                history = (await client.get(f'/history/{cid}', headers=headers)).json()
                assert history['checklist']['_domain_state']['name'] == 'university'
                assert len(history['messages']) == 2
                assert (await client.post(f'/chat/{cid}/action/form', headers=headers)).status_code == 400
                assert (await client.post(f'/chat/{cid}/action/mcp', headers=headers)).status_code == 400
                assert (await client.post('/start-conversation', json={'domain': 'absent'}, headers=headers)).status_code == 400
        finally:
            main.app.dependency_overrides.clear()
            await engine.dispose()
    asyncio.run(run())


def test_streamlit_settings_and_status_render_and_publish(roots, monkeypatch):
    two_domains(roots)
    import domain_admin_ui as ui
    monkeypatch.setattr(ui, 'ROOT_DIR', roots)
    from streamlit.testing.v1 import AppTest
    script = roots / 'admin_smoke.py'
    script.write_text('from domain_admin_ui import render_domain_admin\nrender_domain_admin()\n')
    app = AppTest.from_file(str(script)).run(timeout=15)
    assert not app.exception
    assert any(x.label == 'Gemini: tạo cấu hình tự động' for x in app.button) is False  # No file yet.
    assert any(x.label == 'Áp dụng cấu hình' for x in app.button)
    assert any(x.label == 'Kích hoạt phiên bản đã chọn' for x in app.button)
    description = next(x for x in app.text_area if x.label.startswith('Mô tả phạm vi'))
    description.set_value('Reviewed routing coverage')
    next(x for x in app.button if x.label == 'Áp dụng cấu hình').click()
    app.run(timeout=15)
    assert not app.exception
    assert dc.load_domain_config('legal', root=roots).profile['description'] == 'Reviewed routing coverage'


def test_admin_upload_gemini_review_to_ingestion_button(roots, monkeypatch):
    two_domains(roots)
    import domain_admin_ui as ui
    monkeypatch.setattr(ui, 'ROOT_DIR', roots)
    generated = {**profile('campus'), 'embedding_fields': ['topic']}
    calls = []
    monkeypatch.setattr(ui, 'propose_profile', lambda df, domain: generated)
    def started(path, p, **kwargs):
        calls.append((Path(path), p, kwargs))
        return 'test-job'
    monkeypatch.setattr(ui, 'start_job', started)
    class Upload:
        name = 'campus.csv'
        def getvalue(self):
            return b'Topic,Details\nScholarship,GPA 3.5\n'
    monkeypatch.setattr(ui.st, 'file_uploader', lambda label, **kwargs: Upload() if label == 'Dataset dạng bảng' else None)
    from streamlit.testing.v1 import AppTest
    script = roots / 'upload_smoke.py'
    script.write_text('from domain_admin_ui import render_domain_admin\nrender_domain_admin()\n')
    app = AppTest.from_file(str(script)).run(timeout=15)
    assert not app.exception
    next(b for b in app.button if b.label == 'Gemini: tạo cấu hình tự động').click()
    app.run(timeout=15)
    assert not app.exception
    assert any(b.label == 'Bắt đầu ingest' for b in app.button)
    next(b for b in app.button if b.label == 'Bắt đầu ingest').click()
    app.run(timeout=15)
    assert not app.exception and len(calls) == 1
    path, p, _ = calls[0]
    assert path.exists() and p['domain_name'] == 'campus'
    assert p['schema_mapping'] == generated['schema_mapping']


def test_original_monitoring_dashboard_still_renders_with_admin_tab(roots, monkeypatch):
    two_domains(roots)
    import domain_admin_ui as ui
    monkeypatch.setattr(ui, 'ROOT_DIR', roots)
    monkeypatch.chdir(roots)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(PROJECT / 'backend/admin_dashboard.py')).run(timeout=15)
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    assert 'Quản lý lĩnh vực / dữ liệu' in labels
    assert any('Audit Logs' in label for label in labels)
    assert any('Feedback' in label for label in labels)


def test_friendly_history_excludes_previous_domain_segment(roots):
    from types import SimpleNamespace
    messages = [SimpleNamespace(role='user', content='old legal question', sources='[]'),
                SimpleNamespace(role='assistant', content='legal fees', sources=json.dumps(['RAG_DOMAIN:legal', 'RAG_CONTENT:{}'])),
                SimpleNamespace(role='user', content='new university question', sources='[]'),
                SimpleNamespace(role='assistant', content='campus answer', sources=json.dumps(['RAG_DOMAIN:university'])),
                SimpleNamespace(role='user', content='thanks', sources='[]')]
    selected = chat.domain_history(messages, 'university')
    assert [m.content for m in selected] == ['new university question', 'campus answer', 'thanks']
    assert chat.active_procedures(messages, 'old legal procedure', 'legal') == []
