"""Diagnostic coverage with fake HTTP, vector search, reranker and SQL I/O."""
import asyncio
from contextlib import nullcontext
import importlib.util
import json
import sqlite3
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**stubs, name: module}):
        spec.loader.exec_module(module)
    return module


class DiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        httpx = types.ModuleType('httpx')
        httpx.HTTPError = type('HTTPError', (Exception,), {})
        self.requests = []
        self.raw = '["mượn sách thư viện"]'
        self.http_error = False
        owner = self
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, **kwargs):
                owner.requests.append(kwargs['json'])
                if owner.http_error:
                    raise httpx.HTTPError('offline')
                data = {'choices': [{'message': {'content': owner.raw}}]}
                return types.SimpleNamespace(status_code=200, text=json.dumps(data),
                    raise_for_status=lambda: None, json=lambda: data)
        httpx.AsyncClient = Client
        self.prep = load('rag.prep_diagnostic_test', 'rag/query_preparation.py', {'httpx': httpx})
        self.retrieval = load('rag.retrieval_diagnostic_test', 'rag/retrieval_service.py', {})
        self.pipeline = load('rag.pipeline_diagnostic_test', 'rag/query_pipeline.py', {
            'httpx': httpx, 'rag.query_preparation': self.prep, 'rag.retrieval_service': self.retrieval})
        self.events = []
        self.pipeline.rerank_records = lambda question, matches, **kw: matches[:kw['top_k']]
        for module in (self.prep, self.retrieval, self.pipeline):
            module.emit = lambda task, event, **fields: self.events.append((task, event, fields))
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.config = types.SimpleNamespace(name='university_test', generation='gen', revision='rev',
            profile={}, primary_entity='topic', schema={'topic': 'Chủ đề', 'details': 'Nội dung'},
            evidence_fields=['topic', 'details'], minimum_score=0.1,
            vector_dir=Path(self.root.name), embedding_model='fake', db_path=Path(self.root.name)/'records.db',
            table='records', id_column='id',
            retrieval_option=lambda name, env, default: default)

        with sqlite3.connect(self.config.db_path) as db:
            db.execute('CREATE TABLE records (id TEXT, topic TEXT, details TEXT, description TEXT, submission_method TEXT, fee TEXT, processing_time TEXT)')
            db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)', [
                (str(i), title, '3 books, 14 days', 'Actual content ' + str(i), '', '0', '1 day')
                for i, title in enumerate(['Library','Card','Room','Other','Transcript'],1)])

    def event(self, name):
        return [fields for _, event, fields in self.events if event == name][-1]

    async def test_preparation_logs_invalid_output_and_exact_fallback(self):
        self.raw = '{"wrong":"shape"}'
        result = await self.prep.prepare_search_queries('Library?', 'original query', self.config, force=True)
        self.assertEqual(result.queries, ('original query',))
        self.assertEqual(result.status, 'fallback')
        self.assertEqual(self.event('raw_output')['content'], self.raw)
        self.assertEqual(self.event('fallback')['stage'], 'query_validation')
        self.assertEqual(self.event('fallback')['error'], 'Expected one to eight search phrases')
        self.assertEqual(self.event('request')['payload'], self.requests[0])

    async def test_preparation_success_preserves_phrases(self):
        result = await self.prep.prepare_search_queries('Library?', 'original', self.config, force=True)
        self.assertEqual(result.queries, ('mượn sách thư viện',))
        self.assertEqual(result.status, 'prepared')
        self.assertEqual(self.event('prepared')['queries'], result.queries)

    async def test_preparation_disabled_does_not_call_model(self):
        with patch.object(self.prep, 'preparation_enabled', return_value=False):
            result = await self.prep.prepare_search_queries('question', 'original', self.config)
        self.assertEqual(result.status, 'disabled')
        self.assertEqual(self.requests, [])
        self.assertEqual(self.event('disabled')['queries'], ('original',))

    async def test_preparation_network_failure_retains_fallback(self):
        self.http_error = True
        result = await self.prep.prepare_search_queries('q', 'original', self.config, force=True)
        self.assertEqual(result.status, 'fallback')
        self.assertEqual(self.event('fallback')['stage'], 'http_request')

    def configure_retrieval(self, nodes, scores):
        index = types.SimpleNamespace(as_retriever=lambda **kwargs: types.SimpleNamespace(retrieve=lambda query: nodes))
        self.retrieval._index = lambda *args: index
        def score(pairs):
            self.rerank_pairs = pairs
            return scores
        self.retrieval._reranker = lambda: types.SimpleNamespace(compute_score=score)
        self.retrieval.memory_stage = lambda *args, **kwargs: nullcontext()

    async def test_retrieval_logs_scores_duplicates_cutoff_and_limit(self):
        nodes = [types.SimpleNamespace(text='text', metadata={'id': i, 'topic': title}, score=0.7)
                 for i, title in [(1, 'Library'), (1, 'Library duplicate'), (2, 'Card'), (3, 'Room'), (4, 'Other')]]
        self.configure_retrieval(nodes, [4, 3, 2, 1, -10])
        result = self.retrieval.retrieve_and_rerank('library', top_k=2, config=self.config)
        self.assertEqual([r[0] for r in result], [1, 2])
        self.assertIn('3 books, 14 days', self.rerank_pairs[0][1])
        self.assertNotEqual(self.rerank_pairs[0][1], nodes[0].text)
        self.assertEqual([r[2]['topic'] for r in result], ['Library', 'Card'])
        reasons = [r['reason'] for r in self.event('result')['candidates']]
        self.assertEqual(reasons, ['kept', 'duplicate_record', 'kept', 'result_limit_reached', 'below_minimum_score'])
        self.assertEqual(len(self.event('reranked')['candidates']), 5)
        self.assertEqual(self.event('vector_candidates')['candidates'][0]['vector_score'], 0.7)

    async def test_retrieval_empty_candidates_is_explicit(self):
        self.configure_retrieval([], [])
        self.assertEqual(self.retrieval.retrieve_and_rerank('library', config=self.config), [])
        self.assertEqual(self.event('empty')['reason'], 'vector_search_returned_no_candidates')

    async def test_pipeline_empty_retrieval_is_not_record_selection(self):
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('original',), 'fallback'))
        self.pipeline.retrieve_and_rerank = lambda *args, **kwargs: []
        self.pipeline.select_requested_records = AsyncMock()
        result = await self.pipeline.build_evidence_bundle('q', 'original', config=self.config)
        self.assertEqual(result, ('', 0.0, []))
        self.pipeline.select_requested_records.assert_not_called()
        self.assertEqual(self.event('retrieved')['records'], [])
        self.assertEqual(self.event('empty')['preparation_status'], 'fallback')

    async def test_pipeline_model_rejecting_candidates_is_distinguishable(self):
        matches = [(1, .9, {'topic': 'Library'})]
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('original',), 'fallback'))
        self.pipeline.retrieve_and_rerank = lambda *args, **kwargs: matches
        self.pipeline.call_llm_json = AsyncMock(return_value='[]')
        result = await self.pipeline.build_evidence_bundle('q', 'original', config=self.config)
        self.assertEqual(result, ('', 0.0, []))
        self.assertEqual(self.event('retrieved')['records'][0]['id'], 1)
        self.assertEqual(self.event('selected')['retained_ids'], [])
        self.assertEqual(self.event('output')['raw'], '[]')

    async def test_prepared_exact_title_still_requires_question_validation(self):
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('Library',), 'prepared'))
        self.pipeline.retrieve_and_rerank = lambda *args, **kwargs: [(1, .9, {'topic': 'Library'})]
        self.pipeline.select_requested_records = AsyncMock(return_value=[(1, .9, {'topic': 'Library'})])
        self.pipeline.classify_requested_columns = AsyncMock(return_value=['details'])
        self.pipeline.fetch_record_from_sqlite = AsyncMock(return_value={'topic': 'Library', 'details': '3 books, 14 days'})
        context, score, sources = await self.pipeline.build_evidence_bundle('q', 'Library', config=self.config)
        self.assertEqual(json.loads(context), {'Chủ đề': 'Library', 'Nội dung': '3 books, 14 days'})
        self.assertEqual(score, .9)
        self.assertEqual(sources[0], 'Library')
        self.pipeline.select_requested_records.assert_awaited_once()
        self.assertEqual(self.pipeline.select_requested_records.call_args.args[:2], ('q', 'Library'))
        self.assertEqual(self.event('selection_path')['reason'], 'model_selection_against_question')
        self.assertEqual(self.event('complete')['titles'], ['Library'])

    async def test_json_call_logs_raw_model_envelope(self):
        self.raw = '[0]'
        result = await self.pipeline.call_llm_json('candidates')
        self.assertEqual(result, '[0]')
        self.assertIn('choices', self.event('response')['response'])
        self.assertEqual(self.event('request')['payload']['messages'][1]['content'], 'candidates')

    async def test_json_call_failure_retains_original_fallback(self):
        self.http_error = True
        result = await self.pipeline.call_llm_json('candidates', fallback='INVALID')
        self.assertEqual(result, 'INVALID')
        self.assertEqual(self.event('fallback')['output'], 'INVALID')

    async def test_logged_admissions_gap_does_not_discard_other_methods(self):
        self.config.schema = {'topic': 'Chủ đề', 'description': 'Nội dung', 'submission_method': 'Kênh thực hiện'}
        self.config.evidence_fields = list(self.config.schema)
        self.config.profile = {'selection_score_margin': 0.2, 'field_selection_prompt': 'CUSTOM FIELD RULES'}
        matches = [(1, .714, {'topic': 'Tuyển sinh phương thức 1b'}),
                   (2, .423, {'topic': 'Tuyển sinh phương thức 1a'}),
                   (3, .330, {'topic': 'Tuyển sinh phương thức 2'}),
                   (4, .211, {'topic': 'Hồ sơ xét tuyển người nước ngoài phương thức 1b'}),
                   (5, .060, {'topic': 'Nộp lệ phí xét tuyển nguyện vọng'})]
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('HCMUS phương thức xét tuyển',), 'prepared'))
        self.pipeline.retrieve_and_rerank = lambda *args, **kwargs: matches
        self.pipeline.call_llm_json = AsyncMock(side_effect=['[0,1,2]', '["description"]'])
        rows = {i: {'topic': m['topic'], 'description': 'Thông tin ' + str(i)} for i, _, m in matches}
        self.pipeline.fetch_record_from_sqlite = AsyncMock(side_effect=lambda rid, **kw: rows[rid])
        context, score, sources = await self.pipeline.build_evidence_bundle('HCMUS có những phương thức xét tuyển nào?',
            'HCMUS có những phương thức xét tuyển nào?', config=self.config)
        candidates = json.loads(self.pipeline.call_llm_json.call_args_list[0].args[0])['candidates']
        self.assertEqual(len(candidates), 5)
        self.assertEqual([d['Chủ đề'] for d in json.loads(context)['documents']], [m[2]['topic'] for m in matches[:3]])
        self.assertEqual(sources[:3], [m[2]['topic'] for m in matches[:3]])
        self.assertEqual(score, .714)
        paths = [f['reason'] for _, e, f in self.events if e == 'selection_path']
        self.assertEqual(paths, ['model_selection_against_question'])
        field_call = self.pipeline.call_llm_json.call_args_list[1]
        self.assertIn('CUSTOM FIELD RULES', field_call.kwargs['system'])
        self.assertIn('sample_record_values_by_field', json.loads(field_call.args[0]))
        self.assertTrue(candidates[0]['content'])
        self.assertIn('not a substitute for an explanation', field_call.kwargs['system'])
        self.assertIn('explicitly requested', field_call.kwargs['system'])

    async def test_narrow_field_selection_keeps_description_as_supporting_evidence(self):
        self.config.schema = {'topic': 'Chủ đề', 'description': 'Nội dung', 'fee': 'Phí'}
        self.config.evidence_fields = list(self.config.schema)
        self.pipeline.call_llm_json = AsyncMock(return_value='["fee"]')
        selected = await self.pipeline.classify_requested_columns('Phí bao nhiêu?', config=self.config)
        self.assertEqual(selected, ['fee', 'description'])

    async def test_explicit_requested_fields_survive_incomplete_model_selection(self):
        self.config.schema = {'topic': 'Chủ đề', 'fee': 'Phí', 'processing_time': 'Thời gian'}
        self.config.evidence_fields = list(self.config.schema)
        self.pipeline.call_llm_json = AsyncMock(return_value='["fee"]')
        selected = await self.pipeline.classify_requested_columns('Phí và thời gian?', config=self.config)
        self.assertEqual(set(selected), {'fee', 'processing_time'})

    async def test_record_selector_sees_real_content_and_domain_scope(self):
        self.config.profile['description'] = 'Demo student services'
        self.pipeline.call_llm_json = AsyncMock(return_value='[0]')
        matches = [(1, .9, {'topic': 'Library'})]
        selected = await self.pipeline.select_requested_records('How many books?', 'library', matches, self.config)
        self.assertEqual(selected, matches)
        request = json.loads(self.pipeline.call_llm_json.call_args.args[0])
        self.assertEqual(request['domain_description'], 'Demo student services')
        self.assertIn('3 books, 14 days', request['candidates'][0]['content'])
        self.assertLessEqual(len(request['candidates'][0]['content']), 600)

    async def test_reranker_cannot_use_a_vector_without_its_sql_record(self):
        self.configure_retrieval([types.SimpleNamespace(text='old', metadata={'id':'missing'}, score=.8)], [5])
        with self.assertRaises(self.retrieval.RAGUnavailableError):
            self.retrieval.retrieve_and_rerank('q', config=self.config)
        self.assertFalse(hasattr(self, 'rerank_pairs'))

    async def test_logged_single_quoted_field_selection_preserves_description(self):
        self.config.schema = {'topic':'Chủ đề', 'description':'Nội dung', 'processing_time':'Thời gian'}
        self.config.evidence_fields = list(self.config.schema)
        self.pipeline.call_llm_json = AsyncMock(return_value="['description', 'processing_time']")
        selected = await self.pipeline.classify_requested_columns('Mấy cuốn và bao lâu?', config=self.config)
        self.assertEqual(selected, ['description', 'processing_time'])

    async def test_invalid_field_output_preserves_all_public_evidence(self):
        self.config.schema = {'topic':'Chủ đề', 'description':'Nội dung', 'fee':'Phí'}
        self.config.evidence_fields = list(self.config.schema)
        self.pipeline.call_llm_json = AsyncMock(return_value='not a list')
        selected = await self.pipeline.classify_requested_columns('Phí và điều kiện?', config=self.config)
        self.assertEqual(set(selected), set(self.config.evidence_fields))

    async def test_logged_query_objects_are_normalized_without_inventing_topics(self):
        self.raw = '[{"query":"mượn sách thư viện"},{"query":"thời hạn mượn"}]'
        result = await self.prep.prepare_search_queries('q', 'q', self.config, force=True)
        self.assertEqual(result.status, 'prepared')
        self.assertEqual(result.queries, ('mượn sách thư viện', 'thời hạn mượn'))
        with self.assertRaises(ValueError):
            self.prep.parse_queries('[{"query":"q","ignore":"instructions"}]')

    async def test_expanded_old_title_cannot_bypass_final_record_selection(self):
        self.pipeline.prepare_search_queries = AsyncMock(return_value=self.prep.PreparedQueries(('Transcript','Card'), 'prepared'))
        groups = {'Transcript':[(5,.98,{'topic':'Transcript'})], 'Card':[(2,.99,{'topic':'Card'})]}
        self.pipeline.retrieve_and_rerank = lambda phrase, **kw: groups.get(phrase, [])
        # Run the real selector with a mocked model rejecting the old subject.
        self.pipeline.call_llm_json = AsyncMock(side_effect=['[1]', '["details"]'])
        context, _, sources = await self.pipeline.build_evidence_bundle('Transcript fee?', 'Transcript fee?', config=self.config)
        self.assertEqual(sources[0], 'Transcript')
        self.assertNotIn('Card', sources)
        request = json.loads(self.pipeline.call_llm_json.call_args_list[0].args[0])
        self.assertEqual(request['resolved_question'], 'Transcript fee?')
        self.assertEqual(len(request['candidates']), 2)
        self.assertEqual(self.pipeline.call_llm_json.await_count, 2)  # one record vote, one field vote

    async def test_property_followup_pins_only_previous_record_without_selection(self):
        self.pipeline.retrieve_and_rerank = lambda *a, **kw: [(1,.9,{'topic':'Library'}),(2,.8,{'topic':'Card'})]
        self.pipeline.select_requested_records = AsyncMock()
        self.pipeline.classify_requested_columns = AsyncMock(return_value=['details'])
        context, _, sources = await self.pipeline.build_evidence_bundle('How long?', 'Library. How long?', config=self.config, active_procedures=['Library'])
        self.assertEqual(sources[0], 'Library')
        self.assertNotIn('Card', sources)
        self.pipeline.select_requested_records.assert_not_called()

    def answering_module(self):
        cache = types.ModuleType('cachetools')
        cache.TTLCache = lambda **kwargs: {}
        dotenv = types.ModuleType('dotenv')
        dotenv.load_dotenv = lambda *a: None
        return load('answering_diagnostic_test', 'backend/services/llm_service.py', {
            'httpx':self.prep.httpx, 'cachetools':cache, 'dotenv':dotenv,
            'rag.query_pipeline':self.pipeline})

    async def test_direct_renderer_does_not_discard_borrowing_limits(self):
        self.config.profile['exact_field_output'] = True
        self.config.schema = {'topic':'Chủ đề', 'description':'Nội dung', 'processing_time':'Thời gian'}
        self.config.evidence_fields = list(self.config.schema)
        context = json.dumps({'Chủ đề':'Library','Nội dung':'3 cuốn, 14 ngày','Thời gian':'Trong ngày'})
        answering = self.answering_module()
        self.assertIsNone(answering.render_explicit_lookup(context, 'Thời gian?', self.config))

    async def test_direct_renderer_still_serves_complete_narrow_fee_evidence(self):
        self.config.profile['exact_field_output'] = True
        self.config.schema = {'topic':'Chủ đề', 'fee':'Phí'}
        self.config.evidence_fields = list(self.config.schema)
        answering = self.answering_module()
        context = json.dumps({'Chủ đề':'Card','Phí':'20000 đồng'})
        self.assertIn('20000 đồng', answering.render_explicit_lookup(context, 'Phí?', self.config))

    async def test_actual_admissions_content_key_survives_wrong_channel_vote(self):
        self.config.schema = {'topic':'Chủ đề', 'content':'Nội dung', 'submission_method':'Kênh thực hiện'}
        self.config.evidence_fields = list(self.config.schema)
        self.pipeline.call_llm_json = AsyncMock(return_value='["submission_method"]')
        fields = await self.pipeline.classify_requested_columns('HCMUS có những phương thức xét tuyển nào?', config=self.config)
        self.assertIn('content', fields)
        self.assertNotIn('description', fields)

    async def test_cached_generated_answer_retains_actual_evidence_field_names(self):
        answering = self.answering_module()
        self.config.profile['description']='Equipment workshop Luma with two branches'
        self.config.schema = {'topic':'Chủ đề', 'content':'Nội dung'}
        self.config.evidence_fields = list(self.config.schema)
        answering.load_domain_config = lambda *a: self.config
        answering.config_for_mode = lambda config, mode: config
        answering.mode_settings = lambda *a: {}
        answering.emit = lambda task, event='info', **fields: self.events.append((task, event, fields))
        answering.build_evidence_bundle = AsyncMock(return_value=(
            json.dumps({'Chủ đề':'Method 1a','Nội dung':'Tuyển thẳng'}), .9, ['Method 1a']))
        class Response:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            def raise_for_status(self): pass
            async def aiter_lines(self):
                yield 'data: ' + json.dumps({'choices':[{'delta':{'content':'Tuyển thẳng'}}]})
                yield 'data: [DONE]'
        owner = self
        class Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            def stream(self, *a, **kw):
                owner.requests.append(kw['json'])
                return Response()
        onboarding = types.ModuleType('rag.domain_onboarding')
        onboarding.GENERIC_EVIDENCE_PROMPT = 'Use supplied evidence'
        fastapi = types.ModuleType('fastapi')
        fastapi.HTTPException = type('HTTPException', (Exception,), {})
        with patch.object(answering.httpx, 'AsyncClient', Client), patch.dict(sys.modules, {'rag.domain_onboarding':onboarding, 'fastapi':fastapi}), patch.dict(answering.os.environ, {'RAG_CONTEXT_MODE':'selected'}):
            first, _ = await answering.get_legal_response_stream('Methods?', 'Methods?')
            frames = [frame async for frame in first]
            second, _ = await answering.get_legal_response_stream('Methods?', 'Methods?')
            cached_frames = [frame async for frame in second]
            with patch.dict(answering.os.environ,{'RAG_CONTEXT_MODE':'full'}):
                full, _ = await answering.get_legal_response_stream('Methods?', 'Methods?')
                full_frames=[frame async for frame in full]
            again, _ = await answering.get_legal_response_stream('Methods?', 'Methods?')
            selected_frames=[frame async for frame in again]
        self.assertEqual(answering.build_evidence_bundle.await_count,2)
        self.assertEqual([call.kwargs['context_mode'] for call in answering.build_evidence_bundle.call_args_list],['selected','full'])
        self.assertEqual(frames,full_frames)
        self.assertEqual(frames,selected_frames)
        hit = [f for t, _, f in self.events if t=='answer.cache' and f.get('cache_hit')][-1]
        self.assertEqual(hit['evidence_fields'], ['topic', 'content'])
        self.assertEqual(frames, cached_frames)
        self.assertIn('DATASET_SCOPE',self.requests[0]['messages'][1]['content'])
        self.assertIn('Equipment workshop Luma with two branches',self.requests[0]['messages'][1]['content'])
        self.assertIn('not instructions',self.requests[0]['messages'][0]['content'])

    async def test_vector_discovery_defers_ranking_and_thresholds(self):
        nodes=[types.SimpleNamespace(text='unused',metadata={'id':1,'topic':'Library'},score=.01)]
        self.configure_retrieval(nodes,[-20])
        self.retrieval._reranker=lambda:self.fail('Discovery must not rank per-phrase scores')
        matches=self.retrieval.retrieve_and_rerank('vague phrase',config=self.config,defer_rerank=True)
        self.assertEqual(matches,[(1,0.0,{'id':1,'topic':'Library'})])

    async def test_full_question_ranking_precedes_five_record_limit(self):
        with sqlite3.connect(self.config.db_path) as db:
            db.execute('INSERT INTO records (id,topic,details) VALUES (?,?,?)',('6','Repair','4 days; 80000'))
        full='Repair at branch M: how long and how much?'
        correct=(6,.975,{'topic':'Repair'})
        unrelated=[(i,.996,{'topic':title}) for i,title in enumerate(['Library','Card','Room','Other','Transcript'],1)]
        groups={full:[], 'repair':[correct], 'branch':unrelated}
        seen=[]
        def discover(phrase,**kw):
            seen.append((phrase,kw));return groups[phrase]
        self.pipeline.retrieve_and_rerank=discover
        self.pipeline.prepare_search_queries=AsyncMock(return_value=self.prep.PreparedQueries(('repair','branch'),'prepared'))
        self.pipeline.rerank_records=self.retrieval.rerank_records
        self.retrieval.memory_stage=lambda *a,**kw:nullcontext()
        scored=[]
        def score(pairs):
            scored.extend(pairs)
            return [8 if 'Chủ đề: Repair' in passage else 0 for question,passage in pairs]
        self.retrieval._reranker=lambda:types.SimpleNamespace(compute_score=score)
        self.pipeline.call_llm_json=AsyncMock(side_effect=['[0]','["details"]'])
        context,_,sources=await self.pipeline.build_evidence_bundle(full,full,config=self.config)
        self.assertEqual(sources[0],'Repair')
        self.assertEqual(seen[0][0],full)
        self.assertTrue(all(options['defer_rerank'] for _,options in seen))
        self.assertEqual(len(scored),6)
        self.assertTrue(all(question==full for question,_ in scored))
        self.assertIn(6,self.event('complete')['retained_ids'])
        self.assertLessEqual(len(json.loads(self.pipeline.call_llm_json.call_args_list[0].args[0])['candidates']),5)

    async def test_selector_receives_accent_folded_question_and_titles(self):
        self.pipeline.call_llm_json=AsyncMock(return_value='[0]')
        await self.pipeline.select_requested_records('tram nang muon may khoan','tram nang muon may khoan',
            [(1,.9,{'topic':'Mượn máy khoan — Trạm Nắng'})],self.config)
        payload=json.loads(self.pipeline.call_llm_json.call_args.args[0])
        self.assertIn('tram nang',payload['normalized_question'])
        self.assertIn('tram nang',payload['candidates'][0]['normalized_title'])


if __name__ == '__main__':
    unittest.main()
