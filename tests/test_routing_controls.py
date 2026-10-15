"""Focused routing controls: real temporary domain SQL; classifier/model responses mocked."""
import asyncio
import json
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from test_domain_increment import (roots, two_domains, fake_async_client, routing, chat,
    ChatRequest, Base, db_service, dc)


def run(state=None, requested='auto', **kwargs):
    return asyncio.run(routing.choose_domain('Scholarship requirements', state or {}, [], requested, **kwargs))


def no_detection(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError('Locked routing must skip local and LLM classification')
    monkeypatch.setattr(routing, 'local_domain_matches', fail)
    monkeypatch.setattr(routing.httpx, 'AsyncClient', fail)


@pytest.mark.parametrize('state', [{'name': 'legal', 'selection': 'auto'},
    {'name': 'legal', 'selection': 'auto', 'allow_switching': False, 'locked': True}])
def test_detect_once_skips_all_detection(roots, monkeypatch, state):
    two_domains(roots)
    no_detection(monkeypatch)
    config, saved, changed = run(state)
    assert config.name == 'legal' and not changed
    assert saved['allow_switching'] is False and saved['locked'] is True


def test_first_detection_then_lock(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'university'})
    config, state, changed = run()
    assert config.name == 'university' and changed and state['locked']
    no_detection(monkeypatch)
    assert run(state)[0].name == 'university'


def test_switching_can_use_classifier(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'university'})
    config, state, changed = run({'name': 'legal', 'selection': 'auto', 'allow_switching': True})
    assert config.name == 'university' and changed and state['allow_switching']


def test_switching_can_use_local_match(roots, monkeypatch):
    two_domains(roots)
    monkeypatch.setattr(routing, 'local_domain_matches', lambda *a: ['university'])
    monkeypatch.setattr(routing.httpx, 'AsyncClient', lambda **k: pytest.fail('Unexpected LLM'))
    assert run({'name': 'legal', 'allow_switching': True})[0].name == 'university'


def test_turning_switching_off_keeps_current(roots, monkeypatch):
    two_domains(roots)
    no_detection(monkeypatch)
    config, state, changed = run({'name': 'legal', 'allow_switching': True}, allow_switching=False)
    assert config.name == 'legal' and not changed and not state['allow_switching']


def test_manual_remains_locked_with_switching_on(roots, monkeypatch):
    two_domains(roots)
    no_detection(monkeypatch)
    config, state, _ = run({'name': 'legal', 'selection': 'manual', 'allow_switching': True}, requested=None)
    assert config.name == 'legal' and state['selection'] == 'manual'
    assert run(state, requested='university')[0].name == 'university'


def test_manual_to_auto_redetects(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'university'})
    config, state, changed = run({'name': 'legal', 'selection': 'manual'}, allow_switching=True)
    assert config.name == 'university' and changed and state['selection'] == 'auto'


@pytest.mark.parametrize('kind', ['conversation', 'none'])
def test_non_domain_turn_does_not_lock(roots, monkeypatch, kind):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': kind, 'domain': None})
    _, state, _ = run()
    assert state['locked'] is False
    monkeypatch.setattr(routing, 'local_domain_matches', lambda *a: ['university'])
    assert run(state)[1]['locked'] is True


def test_disabled_lock_is_reselected(roots, monkeypatch):
    two_domains(roots)
    from rag.domain_management import set_domain_enabled
    set_domain_enabled('university', False, root=roots)
    config, state, changed = run({'name': 'university', 'selection': 'auto'})
    assert config.name == 'legal' and changed
    with pytest.raises(HTTPException) as error:
        run(requested='university')
    assert error.value.status_code == 400


def test_classifier_failure_still_clarifies(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'unknown'})
    with pytest.raises(routing.DomainClarification):
        run()


def test_property_followup_with_switching_skips_llm(roots, monkeypatch):
    two_domains(roots)
    no_detection(monkeypatch)
    config, state, changed = asyncio.run(routing.choose_domain('lệ phí bao nhiêu?',
        {'name': 'legal', 'allow_switching': True}, ['Thủ tục đăng ký kết hôn'], 'auto'))
    assert config.name == 'legal' and not changed and state['allow_switching']


def test_reset_and_preference_survive_clarification_in_database(roots, monkeypatch):
    two_domains(roots)
    fake_async_client(monkeypatch, {'kind': 'ambiguous', 'domain': None})
    async def exercise():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'controls.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            conv = await db_service.create_conversation(db, 'controls-user')
            await db_service.update_domain_state(db, conv.id, {'name': 'legal', 'selection': 'auto'})
            await db_service.update_conversation_state(db, conv.id, current_procedure='old title')
            response = await chat.orchestrate_chat(conv.id,
                ChatRequest(user_prompt='unclear topic', domain='auto', reset_domain=True, allow_switching=True), db)
            output = ''.join([chunk async for chunk in response.body_iterator])
            assert 'domain_choices' in output
            await db.refresh(conv)
            assert conv.checklist['_domain_state'] == {'allow_switching': True}
            assert conv.current_procedure is None
        await engine.dispose()
    asyncio.run(exercise())


def test_preference_persists_across_chat_turns(roots, monkeypatch):
    two_domains(roots)
    seen = []
    async def answer(*a, **kw):
        seen.append(kw['domain'])
        async def stream():
            yield 'test answer'
        return stream(), 0.99
    monkeypatch.setattr(chat, 'get_legal_response_stream', answer)
    async def exercise():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'persist.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            conv = await db_service.create_conversation(db, 'controls-user')
            await db_service.update_domain_state(db, conv.id, {'name': 'legal', 'selection': 'auto'})
            for kwargs in ({'allow_switching': False}, {}):
                response = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='Scholarship requirements', domain='auto', **kwargs), db)
                _ = ''.join([chunk async for chunk in response.body_iterator])
            assert seen == ['legal', 'legal']
            fake_async_client(monkeypatch, {'kind': 'domain', 'domain': 'university'})
            response = await chat.orchestrate_chat(conv.id, ChatRequest(user_prompt='Scholarship requirements', domain='auto', allow_switching=True), db)
            _ = ''.join([chunk async for chunk in response.body_iterator])
            await db.refresh(conv)
            assert seen[-1] == 'university'
            assert conv.checklist['_domain_state']['allow_switching'] is True
        await engine.dispose()
    asyncio.run(exercise())

@pytest.mark.parametrize('switching', [False, True])
def test_start_conversation_saves_preference_and_history(roots, monkeypatch, switching):
    two_domains(roots)
    from routers.chat import start_conversation, get_chat_history
    from schemas import StartConversationRequest
    async def exercise():
        engine = create_async_engine('sqlite+aiosqlite:///' + str(roots / 'start.db'))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as db:
            response = await start_conversation(StartConversationRequest(allow_switching=switching), db, 'controls-user')
            cid = response['conversation_id']
        async with sessions() as db:
            history = await get_chat_history(cid, db, 'controls-user')
            assert history.checklist['_domain_state'] == {'allow_switching': switching}
        await engine.dispose()
    asyncio.run(exercise())
