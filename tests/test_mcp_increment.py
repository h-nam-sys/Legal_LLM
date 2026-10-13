"""Focused checks: protocol, evidence filtering, permission, admin, and endpoint behavior."""
import asyncio
import copy
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]
from rag.mcp_config import (validate_settings, load_settings, save_settings, valid_source_url,
                            search_enabled, rule_for, MCPConfigError)
from backend.services import mcp_service as ms
from services.mcp_permission import authorize_pending, SearchLocks
from fastapi import HTTPException


def campus():
    return NS(name='campus_demo', primary_entity='topic', profile={})

def enabled():
    return validate_settings({'enabled': True, 'domain_rules': {'campus_demo': {
        'enabled': True, 'allowed_hosts': ['example.edu'], 'use_location': False}}})

@pytest.mark.parametrize('changes', [
    {'enabled': 'yes'}, {'timeout_seconds': 0}, {'result_limit': 100}, {'max_tokens': True},
    {'transport': 'shell'}, {'transport': 'http', 'server_url': 'http://remote.example/mcp'},
    {'transport': 'http', 'server_url': 'https://user:secret@remote.example/mcp'},
    {'transport': 'http', 'server_url': 'https://remote.example/mcp?key=secret'},
    {'domain_rules': {'campus': {'enabled': True, 'allowed_hosts': ['https://example.edu'], 'use_location': False}}},
    {'extra_arguments': {'query': 'overwrite'}}, {'answer_prompt': ''}])
def test_reject_invalid_settings(changes):
    with pytest.raises(MCPConfigError):
        validate_settings(changes)


def test_settings_are_atomic_and_hot_loaded(tmp_path):
    assert not load_settings(root=tmp_path)['enabled']
    save_settings(enabled(), root=tmp_path)
    bad = {**enabled(), 'result_limit': 0}
    with pytest.raises(MCPConfigError):
        save_settings(bad, root=tmp_path)
    assert load_settings(root=tmp_path) == enabled()
    save_settings({**enabled(), 'enabled': False}, root=tmp_path)
    assert not search_enabled(campus(), load_settings(root=tmp_path))

@pytest.mark.parametrize('url,expected', [
    ('https://example.edu/a', True), ('https://library.example.edu/a', True),
    ('https://example.edu.evil.com/a', False), ('https://evil-example.edu/a', False),
    ('javascript:alert(1)', False), ('https://example.edu@evil.com/', False),
    ('https://example.edu/a\n<script>', False)])
def test_source_host_filter(url, expected):
    assert valid_source_url(url, ['example.edu']) is expected


def test_normalize_requires_real_evidence():
    rows = [{'title': 'Library', 'url': 'https://example.edu/a', 'snippet': 'Borrow with a card'},
            {'title': 'Duplicate', 'url': 'https://example.edu/a', 'snippet': 'Duplicate'},
            {'title': 'Wrong host', 'url': 'https://evil.com/', 'snippet': 'Wrong'},
            {'url': 'https://example.edu/empty', 'snippet': ''}]
    result = NS(isError=False, structuredContent={'results': rows})
    assert len(ms.normalize_results(result, enabled(), ['example.edu'])) == 1
    text = NS(isError=False, structuredContent=None, content=[NS(type='text', text=json.dumps({'results': rows}))])
    assert ms.normalize_results(text, enabled(), ['example.edu']) == ms.normalize_results(result, enabled(), ['example.edu'])
    with pytest.raises(ms.MCPSearchError):
        ms.normalize_results(NS(isError=True), enabled(), [])
    with pytest.raises(ms.MCPSearchError):
        ms.normalize_results(NS(isError=False, structuredContent={'answer': 'Invented'}), enabled(), [])


def test_pending_offer_owner_and_message_id():
    pending = {'domain': 'campus_demo', 'query': 'Borrow?', 'message_id': 20}
    conv = NS(needs_search_permission=True, checklist={'_domain_state': {'name': 'campus_demo'}, '_mcp_pending': pending})
    assert authorize_pending(conv, 'me', 'me', 20) == pending
    for user, offer in [('other', 20), ('me', 19)]:
        with pytest.raises(HTTPException):
            authorize_pending(conv, user, 'me', offer)
    conv.needs_search_permission = False
    with pytest.raises(HTTPException):
        authorize_pending(conv, 'me', 'me', 20)


def test_duplicate_calls_serialize_without_leaking_locks():
    async def run():
        locks = SearchLocks()
        order = []
        async def job(n):
            async with locks.hold(1):
                order.append(n)
                await asyncio.sleep(0)
                order.append(n)
        await asyncio.gather(job(1), job(2))
        assert order == [1, 1, 2, 2]
        assert not locks.entries
    asyncio.run(run())


def test_real_bundled_mcp_discovery_and_missing_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    tools = asyncio.run(ms.inspect_tools())
    assert 'web_search' in {t['name'] for t in tools}
    settings = enabled()
    with pytest.raises(ms.MCPSearchError):
        asyncio.run(ms.retrieve_external('Borrow?', campus(), settings=settings))


@pytest.fixture
def protocol_server(tmp_path, monkeypatch):
    # Real stdio MCP subprocess, real tool and parser; only the Gemini grounding response is substituted.
    script = tmp_path / 'backend/services/mcp_search_server.py'
    script.parent.mkdir(parents=True)
    script.write_text('''import sys, os
sys.path.insert(0, %r)
from backend.services import mcp_search_server as impl
from types import SimpleNamespace as NS
os.environ['GEMINI_API_KEY'] = 'test-only'
async def response(query, model):
    assert 'site:example.edu' in query
    assert model == 'gemini-2.5-flash'
    meta = NS(web_search_queries=[query], grounding_chunks=[
        NS(web=NS(uri='https://example.edu/library', title='Library')),
        NS(web=NS(uri='https://other.edu/library', title='Other'))],
        grounding_supports=[
            NS(segment=NS(text='Use your student card'), grounding_chunk_indices=[0]),
            NS(segment=NS(text='Wrong domain'), grounding_chunk_indices=[1])],
        search_entry_point=NS(rendered_content='<a href="https://www.google.com/search?q=library" target="_blank">Google Search</a>'))
    return NS(candidates=[NS(grounding_metadata=meta)])
impl.generate_grounded = response
impl.server.run(transport='stdio')
''' % str(ROOT), encoding='utf-8')
    monkeypatch.setattr(ms, 'ROOT', tmp_path)
    return tmp_path


def test_real_stdio_search_to_evidence_to_answer(protocol_server, monkeypatch):
    settings = enabled()
    # MCP protocol is live. Substitute only the Qwen HTTP response in the parent.
    original = httpx.AsyncClient
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        evidence = json.loads(payload['messages'][1]['content'])['evidence']
        assert len(evidence) == 1
        assert evidence[0]['url'] == 'https://example.edu/library'
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Bring your student card [1].'}}]})
    monkeypatch.setattr(ms.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    answer, sources = asyncio.run(ms.execute_mcp_search('Borrow books?', config=campus(), settings=settings))
    assert answer == 'Bring your student card [1].'
    assert len(sources) == 2 and 'https://example.edu/library' in sources[0]
    assert sources[1].startswith('RAG_GOOGLE_SEARCH:')
    assert 'Tóm tắt Gemini có dẫn nguồn' in sources[0]
    assert 'Use your student card' in sources[0] and 'Other' not in sources[0]
    assert seen[0]['stream'] is False


def test_disabled_does_not_connect(monkeypatch):
    async def forbid(*a, **k):
        raise AssertionError('Must not connect')
    monkeypatch.setattr(ms, 'inspect_tools', forbid)
    with pytest.raises(ms.MCPSearchError):
        asyncio.run(ms.retrieve_external('Borrow?', campus(), settings=validate_settings({})))


def test_no_results_skips_generation(monkeypatch):
    async def empty(*a, **k):
        return []
    monkeypatch.setattr(ms, 'retrieve_external', empty)
    def forbid(*a, **k):
        raise AssertionError('No model call without evidence')
    monkeypatch.setattr(ms.httpx, 'AsyncClient', forbid)
    answer, sources = asyncio.run(ms.execute_mcp_search('Borrow?', config=campus(), settings=enabled()))
    assert not sources and 'Chưa đủ bằng chứng' in answer


def test_admin_renders_saves_and_keeps_advanced_collapsed(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / 'admin_test.py'
    script.write_text('''import sys
sys.path[:0] = [%r, %r]
import streamlit as st
from types import SimpleNamespace as NS
from unittest.mock import patch
from backend import mcp_admin_ui as ui
from rag.mcp_config import load_settings, save_settings
cfg = NS(name='campus_demo', primary_entity='topic', profile={})
with patch.object(ui, 'list_domain_configs', lambda: [cfg]), patch.object(ui, 'load_settings', lambda: load_settings(root=%r)), patch.object(ui, 'save_settings', lambda value: save_settings(value, root=%r)):
    ui.render_mcp_admin()
''' % (str(ROOT), str(ROOT / 'backend'), str(tmp_path), str(tmp_path)), encoding='utf-8')
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    assert any(e.label == 'Cấu hình nâng cao' and not e.proto.expanded for e in app.expander)
    app.toggle(key='mcp_enabled').set_value(True)
    app.multiselect(key='mcp_domains').set_value(['campus_demo']).run()
    app.text_area(key='mcp_sites_campus_demo').set_value('example.edu')
    next(b for b in app.button if b.label == 'Lưu cấu hình MCP').click().run()
    assert not app.exception
    saved = load_settings(root=tmp_path)
    assert saved['enabled'] and saved['domain_rules']['campus_demo']['allowed_hosts'] == ['example.edu']

@pytest.mark.parametrize('scenario', ['success', 'tool_failure', 'new_question'])
def test_chat_offer_and_action_with_real_sqlite(tmp_path, monkeypatch, scenario):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from database import Base
    from schemas import ChatRequest, MCPActionRequest
    from services import chat_orchestrator as co, db_service as ds
    from routers import chat as routes
    import rag.domain_config as dc
    config = campus()
    async def choose(*a, **k):
        return config, {'name': config.name, 'selection': 'auto'}, False
    async def retrieve(*a, **k):
        async def stream():
            yield 'data: [DONE]\n\n'
        return stream(), 0.0
    monkeypatch.setattr(co, 'choose_domain', choose)
    monkeypatch.setattr(co, 'resolve_mode', lambda *a: 'strict')
    monkeypatch.setattr(co, 'resolve_search_query', lambda prompt, *a: (prompt, []))
    monkeypatch.setattr(co, 'get_legal_response_stream', retrieve)
    monkeypatch.setattr(co, 'search_enabled', lambda c: True)
    monkeypatch.setattr(dc, 'load_domain_config', lambda *a: config)
    monkeypatch.setattr(routes, 'load_settings', enabled)
    calls = []
    async def search(query, *a, **k):
        calls.append(query)
        if scenario == 'tool_failure':
            raise routes.MCPSearchError('Provider unavailable')
        return 'Bring a card [1]', ['[1] Library\nhttps://example.edu/library\nBring a card']
    monkeypatch.setattr(routes, 'execute_mcp_search', search)
    async def run():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path / 'chat.db'))
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as db:
            conv = await ds.create_conversation(db, 'owner', 'MCP integration')
            response = await co.orchestrate_chat(conv.id, ChatRequest(user_prompt='Borrow books?'), db)
            events = ''.join([chunk async for chunk in response.body_iterator])
            assert 'needs_search_permission' in events and not calls
            pending = dict(conv.checklist['_mcp_pending'])
            offer = MCPActionRequest(offer_message_id=pending['message_id'])
            if scenario == 'new_question':
                response = await co.orchestrate_chat(conv.id, ChatRequest(user_prompt='Another question?'), db)
                _ = [chunk async for chunk in response.body_iterator]
                with pytest.raises(HTTPException) as exc:
                    await routes.action_mcp(conv.id, offer, db, 'owner')
                assert exc.value.status_code == 409 and not calls
            elif scenario == 'tool_failure':
                with pytest.raises(HTTPException) as exc:
                    await routes.action_mcp(conv.id, offer, db, 'owner')
                assert exc.value.status_code == 503
                assert conv.needs_search_permission and conv.checklist['_mcp_pending'] == pending
            else:
                result = await routes.action_mcp(conv.id, offer, db, 'owner')
                assert result['text'] == 'Bring a card [1]'
                assert 'RAG_DOMAIN:campus_demo' in result['sources']
                assert not conv.needs_search_permission and '_mcp_pending' not in conv.checklist
                with pytest.raises(HTTPException):
                    await routes.action_mcp(conv.id, offer, db, 'owner')
                assert len(calls) == 1
                messages = await ds.get_conversation_history(db, conv.id)
                assert messages[-1].content == result['text']
                assert 'https://example.edu/library' in messages[-1].sources
        await engine.dispose()
    asyncio.run(run())


def test_real_streamable_http_search(protocol_server, monkeypatch):
    import os
    import socket
    import subprocess
    import time
    for key in ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy']:
        monkeypatch.delenv(key, raising=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    script = protocol_server / 'backend/services/mcp_search_server.py'
    content = script.read_text().replace("impl.server.run(transport='stdio')",
        f"impl.server.settings.host='127.0.0.1'\nimpl.server.settings.port={port}\nimpl.server.run(transport='streamable-http')")
    script.write_text(content)
    log_path = protocol_server / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.05)
            settings = validate_settings({**enabled(), 'transport': 'http', 'server_url': f'http://127.0.0.1:{port}/mcp'})
            records = asyncio.run(ms.retrieve_external('Borrow books?', campus(), settings=settings))
            assert len(records) == 1 and records[0]['url'] == 'https://example.edu/library'
        finally:
            process.terminate()
            process.wait(timeout=10)


def grounding_response(queries=True, support=True, mixed=False):
    chunks = [NS(web=NS(uri='https://example.edu/library', title='Library')),
              NS(web=NS(uri='https://other.edu/library', title='Other'))]
    supports = [NS(segment=NS(text='Bring a student card'), grounding_chunk_indices=[0, 1] if mixed else [0])]
    meta = NS(web_search_queries=['library'] if queries else [], grounding_chunks=chunks,
        grounding_supports=supports if support else [], search_entry_point=None)
    return NS(candidates=[NS(grounding_metadata=meta)])

@pytest.mark.parametrize('queries,support,mixed,expected', [(True,True,False,1),(False,True,False,0),(True,False,False,0),(True,True,True,0)])
def test_gemini_requires_search_and_allowed_citation_support(queries, support, mixed, expected):
    from backend.services import mcp_search_server as server
    result = asyncio.run(server.grounded_records(grounding_response(queries,support,mixed), 5, ['example.edu']))
    assert len(result['results']) == expected


def test_google_redirect_is_resolved_without_fetching_destination(monkeypatch):
    from backend.services import mcp_search_server as server
    original = httpx.AsyncClient
    requests = []
    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={'location': 'https://example.edu/library'})
    monkeypatch.setattr(server.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    url = asyncio.run(server.source_url('https://vertexaisearch.cloud.google.com/grounding-api-redirect/id'))
    assert url == 'https://example.edu/library'
    assert len(requests) == 1 and requests[0].startswith('https://vertexaisearch.cloud.google.com/')


def test_real_genai_sdk_enables_google_search(monkeypatch):
    for key in ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy']:
        monkeypatch.delenv(key, raising=False)
    from backend.services import mcp_search_server as server
    from google import genai
    from google.genai import types
    client_class = genai.Client
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only')
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert payload['tools'][0] == {'googleSearch': {}}
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text':'Bring a card'}], 'role':'model'},
            'groundingMetadata': {'webSearchQueries': ['library'],
              'groundingChunks':[{'web': {'uri':'https://example.edu/library','title':'Library'}}],
              'groundingSupports':[{'segment': {'text':'Bring a card', 'startIndex':0, 'endIndex':12}, 'groundingChunkIndices':[0]}]}}]})
    def client(*a, **k):
        k['http_options'] = types.HttpOptions(httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        return client_class(*a, **k)
    monkeypatch.setattr(server.genai, 'Client', client)
    response = asyncio.run(server.generate_grounded('Borrow?', 'gemini-2.5-flash'))
    records = asyncio.run(server.grounded_records(response, 5, ['example.edu']))
    assert seen and records['results'][0]['snippet'] == 'Bring a card'
