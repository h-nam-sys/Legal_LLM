"""Exercise real routing + orchestration with fake I/O, not real model inference."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import test_llm_domain_routing as routing_fixture

ROOT = Path(__file__).resolve().parents[1]


class ContractIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reuse the ready-domain/model HTTP fixture; run the real router below.
        fixture = routing_fixture.RoutingTests('test_birth_registration_reaches_classifier_with_descriptions')
        fixture.setUp()
        self.fixture = fixture
        router = fixture.router
        self.config = types.SimpleNamespace(name='university_test', primary_entity='topic',
            schema={'topic': 'Chủ đề', 'fee': 'Phí'}, evidence_fields=['topic', 'fee'],
            profile={'description': 'Thư viện Demo', 'routing_examples': ['Mượn sách'], 'allow_forms': False})
        fixture.configs.append(self.config)
        self.conversation = types.SimpleNamespace(checklist={'_domain_state': {
            'name': 'administrative_procedures', 'allow_switching': True, 'locked': True}},
            current_procedure='Thủ tục đăng ký khai sinh', location='', needs_search_permission=False)
        self.messages = [types.SimpleNamespace(role='assistant', content='old answer',
            sources=['Thủ tục đăng ký khai sinh', 'RAG_CONTENT:{}', 'RAG_DOMAIN:administrative_procedures'])]
        self.saved = []
        async def save(db, cid, role, content, sources=None):
            msg = types.SimpleNamespace(id=len(self.messages) + 1, role=role, content=content, sources=sources or [])
            self.messages.append(msg)
            self.saved.append(msg)
            return msg
        async def state(db, cid, state, clear_procedure=False):
            self.conversation.checklist['_domain_state'] = state
            if clear_procedure:
                self.conversation.current_procedure = None
        dbmod = types.ModuleType('services.db_service')
        dbmod.get_conversation = AsyncMock(return_value=self.conversation)
        dbmod.add_message = AsyncMock(side_effect=save)
        dbmod.get_conversation_history = AsyncMock(side_effect=lambda *args: self.messages[:])
        dbmod.update_conversation_title = AsyncMock()
        dbmod.update_conversation_state = AsyncMock()
        dbmod.update_domain_state = AsyncMock(side_effect=state)
        dbmod.log_interaction = AsyncMock()
        self.dbmod = dbmod
        llm = types.ModuleType('services.llm_service')
        async def answer(**kwargs):
            async def stream():
                yield 'data: ' + json.dumps({'type': 'metadata', 'sources': ['Mượn sách', 'RAG_CONTENT:{}']}) + '\n\n'
                yield 'data: ' + json.dumps({'type': 'chunk', 'text': '3 cuốn, 14 ngày'}) + '\n\n'
                yield 'data: [DONE]\n\n'
            return stream(), 0.9
        llm.get_legal_response_stream = AsyncMock(side_effect=answer)
        llm.get_friendly_response_stream = AsyncMock()
        self.llm = llm
        policy = types.ModuleType('services.answer_policy')
        policy.resolve_mode = lambda *args: 'strict'
        policy.needs_retrieval = AsyncMock(return_value=True)
        mcp = types.ModuleType('rag.mcp_config')
        mcp.search_enabled = lambda *args: False
        mcp.MCPConfigError = type('MCPConfigError', (Exception,), {})
        fastapi = types.ModuleType('fastapi')
        fastapi.HTTPException = router.HTTPException
        responses = types.ModuleType('fastapi.responses')
        class Response:
            def __init__(self, body_iterator, **kwargs):
                self.body_iterator = body_iterator
        responses.StreamingResponse = Response
        references = types.ModuleType('rag.turn_context')
        references.resolve_references = AsyncMock(side_effect=lambda question, titles, config: question)
        self.references = references.resolve_references
        services = types.ModuleType('services')
        services.__path__ = []
        stubs = {'rag.turn_context':references, 'services': services, 'services.domain_router': router,
                 'services.llm_service': llm, 'services.answer_policy': policy,
                 'services.db_service': dbmod, 'rag.mcp_config': mcp,
                 'fastapi': fastapi, 'fastapi.responses': responses}
        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location('orchestrator_contract_test', ROOT / 'backend/services/chat_orchestrator.py')
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.db = types.SimpleNamespace(commit=AsyncMock())
        self.payload = types.SimpleNamespace(user_prompt='Ở thư viện Demo, mỗi lần được mượn mấy cuốn và trong bao lâu?',
            domain='auto', allow_switching=True, mode='strict', reset_domain=False)
        # These tests inspect orchestration, while diagnostic capture has its own tests.
        self.core = self.module.orchestrate_chat.__wrapped__.__wrapped__

    async def run_turn(self):
        response = await self.core(161, self.payload, self.db)
        return [frame async for frame in response.body_iterator]

    async def test_single_action_switch_retrieves_only_new_domain_without_old_entities(self):
        self.fixture.decision = {'route': 'university_test'}
        # Avoid importing retrieval libraries; only resolve the query for this test.
        self.module.resolve_search_query = lambda prompt, titles, config, **kw: (prompt, [])
        frames = await self.run_turn()
        kwargs = self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(kwargs['domain'], 'university_test')
        self.assertEqual(kwargs['active_procedures'], [])
        self.assertNotIn('khai sinh', kwargs['search_query'])
        self.assertTrue(any('university_test' in frame for frame in frames))
        self.assertEqual(self.saved[-1].content, '3 cuốn, 14 ngày')
        self.assertTrue(self.dbmod.update_domain_state.call_args.kwargs['clear_procedure'])

    async def test_none_skips_retrieval_and_marks_context_boundary(self):
        self.fixture.decision = {'route': 'none'}
        frames = await self.run_turn()
        self.llm.get_legal_response_stream.assert_not_called()
        self.llm.get_friendly_response_stream.assert_not_called()
        self.assertEqual(self.saved[-1].sources, ['RAG_ROUTE:none'])
        self.assertNotIn('name', self.conversation.checklist['_domain_state'])
        self.assertIsNone(self.conversation.current_procedure)
        self.assertEqual(self.module.active_procedures(self.messages, 'old title', 'administrative_procedures'), [])
        self.assertEqual(self.module.domain_history(self.messages, 'administrative_procedures'), [])
        self.assertEqual(self.dbmod.log_interaction.call_args.args[-1], 'DOMAIN_NONE')
        self.assertEqual(frames[-1], 'data: [DONE]\n\n')

    async def test_exhausted_retry_clarifies_without_retrieval(self):
        self.fixture.decision = {'kind': 'followup', 'domain': 'university_test'}
        frames = await self.run_turn()
        self.llm.get_legal_response_stream.assert_not_called()
        self.assertEqual(self.dbmod.log_interaction.call_args.args[-1], 'DOMAIN_CLARIFICATION')
        self.assertEqual(len(self.fixture.requests), 2)
        self.assertTrue(any('domain_choices' in frame for frame in frames))

    async def test_conversation_skips_rag_preserves_subject_and_followup(self):
        self.fixture.decision = {'route': 'conversation'}
        self.payload.user_prompt = 'Cảm ơn bạn!'
        calls = []
        def friendly(prompt, history, config=None):
            calls.append((prompt, history, config))
            async def stream():
                yield 'data: {"type":"metadata","sources":[]}\n\n'
                yield 'data: {"type":"chunk","text":"Rất vui được hỗ trợ bạn!"}\n\n'
                yield 'data: [DONE]\n\n'
            return stream()
        self.module.get_friendly_response_stream = friendly
        frames = await self.run_turn()
        self.llm.get_legal_response_stream.assert_not_called()
        self.assertEqual(self.conversation.checklist['_domain_state']['name'], 'administrative_procedures')
        self.assertEqual(self.conversation.current_procedure, 'Thủ tục đăng ký khai sinh')
        self.assertEqual(calls[0][0], self.payload.user_prompt)
        self.assertIsNone(calls[0][2])
        self.assertEqual(len(calls[0][1]), 1)
        self.assertEqual(calls[0][1][0].role, 'user')
        self.assertEqual(self.saved[-1].sources, ['RAG_ROUTE:conversation'])
        self.assertEqual(self.saved[-1].content, 'Rất vui được hỗ trợ bạn!')
        self.assertEqual(self.dbmod.log_interaction.call_args.args[-1], 'CONVERSATION')
        self.assertEqual(frames[-1], 'data: [DONE]\n\n')
        self.fixture.decision = {'route': 'followup'}
        self.payload.user_prompt = 'Còn trường hợp đó?'
        self.module.resolve_search_query = lambda prompt, titles, config, **kw: (prompt, titles)
        await self.run_turn()
        kwargs = self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(kwargs['domain'], 'administrative_procedures')
        self.assertEqual(kwargs['active_procedures'], ['Thủ tục đăng ký khai sinh'])

    async def test_conversation_before_any_dataset_does_not_lock_one(self):
        self.conversation.checklist = {'_domain_state': {'allow_switching': True}}
        self.conversation.current_procedure = None
        self.messages.clear()
        self.fixture.decision = {'route': 'conversation'}
        self.payload.user_prompt = 'Xin chào'
        async def friendly(*args):
            yield 'data: {"type":"chunk","text":"Xin chào!"}\n\n'
            yield 'data: [DONE]\n\n'
        self.module.get_friendly_response_stream = friendly
        await self.run_turn()
        self.assertNotIn('name', self.conversation.checklist['_domain_state'])
        self.llm.get_legal_response_stream.assert_not_called()
        self.assertEqual(self.saved[-1].content, 'Xin chào!')

    def test_explicit_new_subject_receives_no_old_search_context(self):
        self.module.property_followup = lambda *a: False
        question = 'Xin bảng điểm tại Demo có phí không?'
        query, pinned = self.module.resolve_search_query(question, ['Cấp lại thẻ sinh viên'], self.config)
        self.assertEqual(query, question)
        self.assertEqual(pinned, [])

    def test_action_followup_receives_subject_without_pinning_old_record(self):
        self.module.property_followup = lambda *a: False
        query, pinned = self.module.resolve_search_query('Có được gia hạn không?', ['Mượn sách thư viện'], self.config, followup=True)
        self.assertIn('Mượn sách thư viện', query)
        self.assertEqual(pinned, [])

    def test_property_followup_keeps_the_previous_records(self):
        self.module.property_followup = lambda *a: True
        query, pinned = self.module.resolve_search_query('Nộp ở đâu?', ['Cấp lại thẻ sinh viên'], self.config)
        self.assertEqual(pinned, ['Cấp lại thẻ sinh viên'])
        self.assertIn('Cấp lại thẻ sinh viên', query)

    async def test_same_dataset_route_still_resolves_implicit_subject(self):
        self.fixture.decision = {'route':'university_test'}
        self.conversation.checklist['_domain_state']['name']='university_test'
        self.messages[0].sources=['Mượn máy khoan — Trạm Mây','RAG_CONTENT:{}','RAG_DOMAIN:university_test']
        self.payload.user_prompt='Gia hạn được mấy lần?'
        self.module.resolve_search_query=lambda question,titles,config:(question,[])
        self.references.side_effect=None
        self.references.return_value='CURRENT QUESTION: Gia hạn được mấy lần? REFERENCED SUBJECT: Trạm Mây'
        await self.run_turn()
        self.references.assert_awaited_once_with('Gia hạn được mấy lần?', ['Mượn máy khoan — Trạm Mây'], self.config)
        self.assertIn('Trạm Mây',self.llm.get_legal_response_stream.call_args.kwargs['search_query'])
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['active_procedures'],[])

    async def test_changed_domain_does_not_resolve_old_subject(self):
        self.fixture.decision = {'route':'university_test'}
        self.module.resolve_search_query=lambda question,titles,config:(question,[])
        await self.run_turn()
        self.references.assert_not_awaited()

    async def test_invalid_reference_resolution_does_not_guess_an_answer(self):
        self.fixture.decision = {'route':'administrative_procedures'}
        self.module.resolve_search_query=lambda question,titles,config:(question,[])
        self.references.side_effect=self.module.RAGUnavailableError('invalid indices')
        with self.assertRaises(self.fixture.router.HTTPException) as caught:
            await self.run_turn()
        self.assertEqual(caught.exception.status_code,503)
        self.llm.get_legal_response_stream.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
