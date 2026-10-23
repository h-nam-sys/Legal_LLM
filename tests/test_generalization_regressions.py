"""Behavioral regressions using real orchestration/ranking and mocked external I/O.
No model-quality or latency claims are made by these tests.
"""
import asyncio
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch
import test_pipeline_diagnostics as pipeline_fixture
import test_routing_contract_integration as chat_fixture
import test_llm_domain_routing as router_fixture

ROOT = Path(__file__).resolve().parents[1]


class PendingQuestionTests(unittest.IsolatedAsyncioTestCase):
    setUp = chat_fixture.ContractIntegrationTests.setUp
    run_turn = chat_fixture.ContractIntegrationTests.run_turn

    async def test_clarification_is_saved_then_resumed_without_old_subjects(self):
        self.messages.clear()
        self.conversation.current_procedure = None
        self.conversation.checklist = {'_domain_state': {'allow_switching': True}}
        self.payload.user_prompt = 'How many working days does it take?'
        self.fixture.decision = {'route': 'clarify'}
        await self.run_turn()
        self.assertEqual(self.conversation.checklist['_rag_pending_question'], self.payload.user_prompt)
        self.payload.user_prompt = 'I mean the library membership card.'
        self.fixture.decision = {'route': 'university_test', 'resume_pending': True}
        self.module.resolve_search_query = lambda q, titles, config: (q, titles)
        before = len(self.fixture.requests)
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests) - before, 1)
        args = self.llm.get_legal_response_stream.call_args.kwargs
        self.assertIn('How many working days', args['prompt'])
        self.assertIn('library membership', args['search_query'])
        self.assertEqual(args['active_procedures'], [])
        self.assertNotIn('_rag_pending_question', self.conversation.checklist)
        self.assertNotIn('resume_pending', self.conversation.checklist['_domain_state'])
        self.references.assert_not_awaited()

    async def test_new_standalone_question_discards_pending_intent(self):
        self.conversation.checklist['_rag_pending_question'] = 'How long for a library card?'
        self.payload.user_prompt = 'What documents are required for birth registration?'
        self.fixture.decision = {'route': 'administrative_procedures', 'resume_pending': False}
        self.module.resolve_search_query = lambda q, *args: (q, [])
        self.references.side_effect = lambda q, *args: q
        await self.run_turn()
        args = self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(args['prompt'], self.payload.user_prompt)
        self.assertNotIn('library', args['search_query'])
        self.assertNotIn('_rag_pending_question', self.conversation.checklist)

    async def test_reset_clears_pending_question_before_routing(self):
        self.conversation.checklist['_rag_pending_question'] = 'Old unresolved request'
        self.payload.reset_domain = True
        self.fixture.decision = {'route': 'university_test'}
        self.module.resolve_search_query = lambda q, *args: (q, [])
        await self.run_turn()
        request = json.loads(self.fixture.requests[-1]['messages'][1]['content'])
        self.assertNotIn('pending_question', request)
        self.assertNotIn('_rag_pending_question', self.conversation.checklist)

    async def test_social_interlude_preserves_pending_question(self):
        self.conversation.checklist['_rag_pending_question'] = 'How many days?'
        self.fixture.decision = {'route': 'conversation', 'resume_pending': False}
        self.payload.user_prompt = 'Thanks!'
        async def stream(*args):
            yield 'data: {"type":"chunk","text":"Welcome"}\n\n'
            yield 'data: [DONE]\n\n'
        self.module.get_friendly_response_stream = stream
        await self.run_turn()
        self.assertEqual(self.conversation.checklist['_rag_pending_question'], 'How many days?')
        self.llm.get_legal_response_stream.assert_not_awaited()

    async def test_none_clears_pending_and_old_evidence(self):
        self.conversation.checklist['_rag_pending_question'] = 'How many days?'
        self.fixture.decision = {'route': 'none', 'resume_pending': False}
        await self.run_turn()
        self.assertNotIn('_rag_pending_question', self.conversation.checklist)
        self.assertEqual(self.saved[-1].sources, ['RAG_ROUTE:none'])

    async def test_cache_bypass_is_passed_per_request(self):
        self.fixture.decision = {'route': 'university_test'}
        self.payload.bypass_answer_cache = True
        self.module.resolve_search_query = lambda q, *args: (q, [])
        await self.run_turn()
        self.assertFalse(self.llm.get_legal_response_stream.call_args.kwargs['use_cache'])

    async def test_clarification_change_can_be_disabled_for_ablation(self):
        self.payload.user_prompt = 'How many days?'
        self.fixture.decision = {'route':'clarify'}
        with patch.dict(self.module.enabled.__globals__['os'].environ, {'RAG_RESUME_CLARIFICATION':'0'}):
            await self.run_turn()
        self.assertNotIn('_rag_pending_question',self.conversation.checklist)

    async def test_oversize_question_is_not_truncated_into_a_different_intent(self):
        self.payload.user_prompt = 'x' * 2100 + ' not eligible'
        self.fixture.decision = {'route':'clarify'}
        await self.run_turn()
        self.assertNotIn('_rag_pending_question',self.conversation.checklist)


class PendingContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = router_fixture.RoutingTests()
        self.fixture.setUp()
        self.router = self.fixture.router

    def test_resume_requires_boolean_and_explicit_dataset(self):
        for decision in ({'route':'followup','resume_pending':True},
                         {'route':'university_admission','resume_pending':'true'},
                         {'route':'university_admission'},
                         {'route':'university_admission','resume_pending':True,'answer':'invented'}):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.router.parse_domain_decision(json.dumps(decision), ['university_admission'], pending=True)

    def test_no_pending_keeps_original_single_action_contract(self):
        with self.assertRaises(ValueError):
            self.router.parse_domain_decision('{"route":"none","resume_pending":false}', [])
        self.assertEqual(self.router.parse_domain_decision('{"route":"none"}', []), {'route':'none'})

    def test_invalid_pending_decision_repaired_in_existing_call_budget(self):
        self.fixture.decisions = [{'route':'followup','resume_pending':True},
                                 {'route':'university_admission','resume_pending':True}]
        _, state, _ = asyncio.run(self.router.choose_domain('I mean admissions.', {}, [], 'auto', pending_question='What is the fee?'))
        self.assertTrue(state['resume_pending'])
        self.assertEqual(len(self.fixture.requests), 2)
        request = json.loads(self.fixture.requests[0]['messages'][1]['content'])
        self.assertEqual(request['pending_question'], 'What is the fee?')


class RankingRecoveryTests(unittest.IsolatedAsyncioTestCase):
    setUp = pipeline_fixture.DiagnosticTests.setUp
    event = pipeline_fixture.DiagnosticTests.event
    answering_module = pipeline_fixture.DiagnosticTests.answering_module

    def configure_scores(self, scores):
        self.retrieval.memory_stage = lambda *a, **kw: nullcontext()
        self.pairs = []
        def compute(pairs):
            self.pairs.extend(pairs)
            return scores[:len(pairs)]
        self.retrieval._reranker = lambda: types.SimpleNamespace(compute_score=compute)
        return [(i, 0.0, {'topic':title}) for i, title in enumerate(['Library','Card','Room','Other','Transcript'],1)]

    def test_low_scores_remain_rejected_without_selector_opt_in(self):
        matches = self.configure_scores([-8,-9,-10,-11,-12])
        self.assertEqual(self.retrieval.rerank_records('question',matches,config=self.config,top_k=5), [])

    def test_low_score_recovery_bounded_and_raw_scores_preserved(self):
        matches = self.configure_scores([-8,-9,-10,-11,-12])
        kept = self.retrieval.rerank_records('question',matches,config=self.config,top_k=5,allow_low_score=True)
        self.assertEqual([m[0] for m in kept], [1,2,3])
        self.assertAlmostEqual(kept[0][1], self.retrieval._sigmoid(-8))
        self.assertTrue(all(m[1] < self.config.minimum_score for m in kept))
        self.assertEqual(len(self.pairs), 5)
        self.assertTrue(self.event('complete')['low_score_recovery'])

    def test_high_score_path_does_not_expand_with_weak_records(self):
        matches = self.configure_scores([2,-8,-9,-10,-11])
        kept = self.retrieval.rerank_records('question',matches,config=self.config,top_k=5,allow_low_score=True)
        self.assertEqual([m[0] for m in kept], [1])
        self.assertFalse(self.event('complete')['low_score_recovery'])

    async def build_low_score_bundle(self, decision):
        matches = self.configure_scores([-8,-9,-10,-11,-12])
        self.pipeline.rerank_records = self.retrieval.rerank_records
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('query',),'prepared'))
        self.pipeline.retrieve_and_rerank = lambda *a, **kw: matches
        self.pipeline.call_llm_json = AsyncMock(return_value=decision)
        return await self.pipeline.build_evidence_bundle('query','query',config=self.config,context_mode='full')

    async def test_recovered_pool_still_requires_selector_and_can_abstain(self):
        self.assertEqual(await self.build_low_score_bundle('[]'), ('',0.0,[]))
        self.assertEqual(self.pipeline.call_llm_json.await_count, 1)
        candidates = json.loads(self.pipeline.call_llm_json.call_args.args[0])['candidates']
        self.assertEqual(len(candidates), 3)

    async def test_recovered_record_reaches_full_evidence_with_true_score(self):
        context, score, sources = await self.build_low_score_bundle('[1]')
        self.assertIn('Card', context)
        self.assertLess(score, self.config.minimum_score)
        self.assertEqual(sources[0], 'Card')
        self.assertEqual(self.pipeline.call_llm_json.await_count, 1)

    async def test_original_ascii_query_survives_corrupt_preparation_and_resolution(self):
        original = 'khong co giay 2 ban'
        resolved = 'contextual question'
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('Có giấy 3 bản',),'prepared'))
        seen = []
        self.pipeline.retrieve_and_rerank = lambda phrase, **kw: seen.append(phrase) or []
        await self.pipeline.build_evidence_bundle(original,resolved,config=self.config,context_mode='full')
        self.assertIn(original, seen)
        self.assertIn(resolved, seen)

    async def test_cache_bypass_recomputes_without_reading_or_writing_existing_entry(self):
        answering = self.answering_module()
        answering.load_domain_config = lambda *a: self.config
        answering.config_for_mode = lambda c, m: c
        answering.mode_settings = lambda *a: {}
        # No evidence is an inexpensive way to verify no cache read or write occurs.
        answering.build_evidence_bundle = AsyncMock(return_value=('',0.0,[]))
        class PoisonCache(dict):
            def __contains__(self, key): raise AssertionError('Cache read during bypass')
            def __setitem__(self, key, value): raise AssertionError('Cache write during bypass')
        answering.qa_cache = PoisonCache()
        onboarding = types.ModuleType('rag.domain_onboarding')
        onboarding.GENERIC_EVIDENCE_PROMPT = 'Use evidence'
        fastapi = types.ModuleType('fastapi')
        fastapi.HTTPException = type('HTTPException',(Exception,),{})
        with patch.dict(sys.modules, {'rag.domain_onboarding':onboarding,'fastapi':fastapi}):
            for _ in range(2):
                stream, score = await answering.get_legal_response_stream('same','same',use_cache=False)
                self.assertEqual(score,0.0)
                self.assertEqual([frame async for frame in stream][-1], 'data: [DONE]\n\n')
        self.assertEqual(answering.build_evidence_bundle.await_count,2)

    async def test_low_score_evidence_generates_and_cache_bypass_does_not_write(self):
        answering = self.answering_module()
        self.config.schema = {'topic':'Entity', 'details':'Rules', 'fee':'Price'}
        self.config.evidence_fields = list(self.config.schema)
        self.config.profile['field_aliases'] = {'fee':['cost']}
        answering.load_domain_config = lambda *a: self.config
        answering.config_for_mode = lambda c, m: c
        answering.mode_settings = lambda *a: {}
        answering.build_evidence_bundle = AsyncMock(return_value=(
            json.dumps({'Entity':'Service A','Rules':'Only members; 2 days','Price':'12 per item'}),
            .001, ['Service A', 'RAG_CONTENT:{}']))
        class Response:
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            def raise_for_status(self): pass
            async def aiter_lines(self):
                yield 'data: ' + json.dumps({'choices':[{'delta':{'content':'12 per item'}}]})
                yield 'data: [DONE]'
        requests = []
        class Client:
            def __init__(self,**kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            def stream(self,*args,**kwargs):
                requests.append(kwargs['json'])
                return Response()
        onboarding = types.ModuleType('rag.domain_onboarding')
        onboarding.GENERIC_EVIDENCE_PROMPT = 'Use evidence'
        fastapi = types.ModuleType('fastapi')
        fastapi.HTTPException = type('HTTPException',(Exception,),{})
        with patch.dict(sys.modules, {'rag.domain_onboarding':onboarding,'fastapi':fastapi}), patch.object(answering.httpx,'AsyncClient',Client):
            # Seed an ordinary entry, then ensure bypass neither reads nor replaces it.
            stream, score = await answering.get_legal_response_stream('cost?','cost?')
            first = [frame async for frame in stream]
            self.assertEqual(score,.001)  # Real relevance, not a promoted confidence score.
            self.assertIn('12 per item', ''.join(first))
            self.assertEqual(len(answering.qa_cache),1)
            before = dict(answering.qa_cache)
            for _ in range(2):
                stream, _ = await answering.get_legal_response_stream('cost?','cost?',use_cache=False)
                self.assertEqual([frame async for frame in stream],first)
            self.assertEqual(answering.qa_cache,before)
        self.assertEqual(answering.build_evidence_bundle.await_count,3)
        self.assertEqual(len(requests),3)
        user = requests[0]['messages'][1]['content']
        self.assertIn('REQUESTED_FIELD_HINTS',user)
        self.assertIn('Only members; 2 days',user)  # Full record retained despite narrow request.
        self.assertIn('Price',user)

    async def test_recovery_can_be_disabled_for_ablation(self):
        self.configure_scores([-8,-9,-10,-11,-12])
        self.pipeline.rerank_records = self.retrieval.rerank_records
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('query',),'prepared'))
        self.pipeline.retrieve_and_rerank = lambda *a, **kw: [(1,0.0,{'topic':'Library'})]
        self.pipeline.call_llm_json = AsyncMock(side_effect=AssertionError('Selector called after hard cutoff'))
        import os
        with patch.dict(os.environ, {'RAG_LOW_SCORE_RECOVERY':'0'}):
            result = await self.pipeline.build_evidence_bundle('query','query',config=self.config,context_mode='full')
        self.assertEqual(result, ('',0.0,[]))


class SpeedReportTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('qa_test_metrics',ROOT/'qa_suite.py')
        self.qa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.qa)

    def test_cached_and_uncached_latency_are_reported_separately(self):
        turns = [{'elapsed_ms':v,'request_id':str(i)} for i,v in enumerate([100000,3000,14000,16000])]
        events = [{'task':'answer.cache','request_id':'1','cache_hit':True}]
        metrics = self.qa.performance_summary(turns,events)
        self.assertEqual(metrics['first_request_seconds'],100)
        self.assertEqual(metrics['cache_hits'],1)
        self.assertEqual(metrics['excluding_first_request']['mean_seconds'],11)
        self.assertEqual(metrics['uncached_excluding_first_request']['mean_seconds'],15)

    def test_cacheless_payload_overrides_fixture_option(self):
        class Client:
            def request(self,path,payload=None,**kw):
                if path == '/start-conversation': return {'conversation_id':1},None,1
                if path == '/history/1': return {'messages':[]},None,1
                self.payload = payload
                return {'answer':'ok'},'trace',1
        import tempfile
        client = Client()
        cases = [{'id':'sample','turns':[{'prompt':'q','options':{'bypass_answer_cache':False}}]}]
        with tempfile.TemporaryDirectory() as folder:
            self.qa.execute_suite(client,cases,Path(folder)/'checkpoint.jsonl',cacheless=True)
        self.assertTrue(client.payload['bypass_answer_cache'])

    def test_old_backend_cannot_silently_claim_cacheless_success(self):
        turn = {'request_id':'trace','expected':{},'payload':{'bypass_answer_cache':True},
                'actual':{'stream':{'metadata':{},'done':True,'stream_errors':[],'answer':'ok'},
                          'saved_message':{'content':'ok','sources':[]}}}
        legacy = [{'request_id':'trace','task':'answer.cache','cache_hit':False}]
        status, checks = self.qa.check_turn(turn,legacy)
        self.assertEqual(status,'FAIL')
        self.assertTrue(any(c['name']=='answer_cache_bypassed' and c['status']=='FAIL' for c in checks))
        legacy[0]['cache_enabled'] = False
        self.assertEqual(self.qa.check_turn(turn,legacy)[0], 'PASS')

    def test_comparison_flags_slowdown_and_unverified_cache_conditions(self):
        spec = importlib.util.spec_from_file_location('cacheless_driver_test', ROOT/'testing/run_cacheless_tests.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        run = {'turn_count':48,'counts':{'PASS':44,'FAIL':4},'answer_cache_bypassed':True,
               'performance':{'uncached_excluding_first_request':{'mean_seconds':16},
                              'answer_cache_events':40,'cache_hits':0,'cache_bypass_acknowledged':40}}
        baseline = {'fresh_48':{'automatic_counts':{'PASS':44,'FAIL':4},
                              'uncached_excluding_first_mean_seconds':14}}
        compared = driver.compare_run('fresh_48',run,baseline)
        self.assertTrue(compared['cacheless_verified'])
        self.assertTrue(compared['speed_review_required'])
        run['performance']['cache_bypass_acknowledged'] = 39
        self.assertFalse(driver.compare_run('fresh_48',run,baseline)['cacheless_verified'])


if __name__ == '__main__':
    unittest.main()
