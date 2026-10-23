"""Representation, grading and controlled-comparison behavior; no live model claims."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch
import test_pipeline_diagnostics as fixture
import test_routing_contract_integration as chat_fixture

ROOT = Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnitCheckerTests(unittest.TestCase):
    def setUp(self): self.qa=load('unit_checker_test','qa_suite.py')

    def test_equivalent_explicit_unit_wordings(self):
        cases=[('Tại Demo, một bản xin bảng điểm là 10000 đồng.','ban'),
               ('Phí cho từng thẻ là 20000 đồng.','the'),
               ('Giá là 12.000đ/bản.','ban'),
               ('Sửa một máy hết 80000 đồng.','may'),
               ('Giá cho 1 vé là 50000 VND.','ve')]
        for text,unit in cases:
            with self.subTest(text=text):
                self.assertRegex(self.qa.unit_equivalent_text(text),r'\bmoi '+unit+r'\b')

    def test_does_not_infer_units_across_negation_or_different_explicit_rates(self):
        cases=[('Một bản không có giá 10000 đồng.','ban'),
               ('Một thẻ có giá 10000 đồng mỗi tháng.','the'),
               ('Một máy. Phí đăng ký là 10000 đồng.','may'),
               ('Một máy\nPhí đăng ký là 10000 đồng.','may'),
               ('Không có thông tin về giá.','the'),
               ('Phí là 10000 đồng.','ban')]
        for text,unit in cases:
            with self.subTest(text=text):
                self.assertNotRegex(self.qa.unit_equivalent_text(text),r'\bmoi '+unit+r'\b')

    def turn(self,answer):
        return {'request_id':'r','expected':{'answer_patterns':['moi ban'],'require_answer_patterns':True},
                'actual':{'stream':{'metadata':{},'done':True,'stream_errors':[],'answer':answer},
                          'saved_message':{'content':answer,'sources':[]}}}

    def test_old_and_new_checker_results_remain_reproducible(self):
        row=self.turn('Một bản xin bảng điểm là 10000 đồng.')
        self.assertEqual(self.qa.check_turn(row,[],unit_equivalence=False)[0],'FAIL')
        self.assertEqual(self.qa.check_turn(row,[])[0],'PASS')
        self.assertEqual(row['actual']['stream']['answer'],'Một bản xin bảng điểm là 10000 đồng.')

    def test_guidance_override_must_be_acknowledged(self):
        row=self.turn('10000 đồng mỗi bản.');row['payload']={'answer_guidance':False}
        events=[{'request_id':'r','task':'answer.cache','runtime_options':{'RAG_ANSWER_GUIDANCE':True}}]
        self.assertEqual(self.qa.check_turn(row,events)[0],'FAIL')
        events[0]['runtime_options']['RAG_ANSWER_GUIDANCE']=False
        self.assertEqual(self.qa.check_turn(row,events)[0],'PASS')


class SelectorViewTests(unittest.IsolatedAsyncioTestCase):
    setUp=fixture.DiagnosticTests.setUp

    async def test_restored_selector_keeps_original_evidence_and_normalized_matching_hints(self):
        import sqlite3
        with sqlite3.connect(self.config.db_path) as db:
            db.execute('UPDATE records SET details=? WHERE id=?',('Không cho gia hạn. Tối đa 2 ngày; 12000 đồng mỗi lượt.','1'))
        self.config.profile['description']='Dịch vụ thử nghiệm tại Trạm Nắng'
        self.config.schema={'topic':'Sản phẩm','details':'Điều kiện và hạn mức'}
        self.pipeline.call_llm_json=AsyncMock(return_value='[0]')
        result=await self.pipeline.select_requested_records('tram nang co duoc gia han khong?',
            'tram nang co duoc gia han khong?',[(1,.001,{'topic':'Dịch vụ — Trạm Nắng'})],self.config)
        request=json.loads(self.pipeline.call_llm_json.call_args.args[0])
        content=request['candidates'][0]['content']
        self.assertNotIn('content_view',request)
        self.assertIn('Không cho gia hạn',content)
        self.assertIn('2 ngày',content);self.assertIn('12000 đồng mỗi lượt',content)
        self.assertIn('Trạm Nắng',request['domain_description'])
        self.assertIn('tram nang',request['candidates'][0]['normalized_title'])
        self.assertLessEqual(len(content),500)
        self.assertEqual(result[0][1],.001)
        # Source row used later for answering retains its original accented values.
        row=await self.pipeline.fetch_record_from_sqlite(1,config=self.config)
        self.assertIn('Không cho gia hạn',row['details'])

    async def test_accented_questions_keep_original_evidence_view(self):
        self.pipeline.call_llm_json=AsyncMock(return_value='[0]')
        await self.pipeline.select_requested_records('Thư viện cho mượn bao lâu?','Thư viện cho mượn bao lâu?',
            [(1,.9,{'topic':'Thư viện'})],self.config)
        request=json.loads(self.pipeline.call_llm_json.call_args.args[0])
        self.assertNotIn('content_view',request)
        self.assertIn('Chủ đề:',request['candidates'][0]['content'])

    async def test_valid_empty_selection_is_not_overridden_or_retried(self):
        self.pipeline.call_llm_json=AsyncMock(return_value='[]')
        result=await self.pipeline.select_requested_records('unrelated request','unrelated request',
            [(1,.9,{'topic':'Library'})],self.config)
        self.assertEqual(result,[])
        self.assertEqual(self.pipeline.call_llm_json.await_count,1)

    async def test_discarded_evidence_folding_is_not_reenabled_by_query_guard(self):
        import os
        self.pipeline.call_llm_json=AsyncMock(return_value='[]')
        requests=[]
        for value in ('0','1'):
            with patch.dict(os.environ,{'RAG_ASCII_QUERY_GUARD':value}):
                await self.pipeline.select_requested_records('library?','library?',[(1,.9,{'topic':'Library'})],self.config)
            requests.append(json.loads(self.pipeline.call_llm_json.call_args.args[0]))
        self.assertEqual(requests[0],requests[1])
        self.assertIn('Chủ đề:',requests[0]['candidates'][0]['content'])
        self.assertNotIn('content_view',requests[0])


class GuidanceTransportTests(unittest.IsolatedAsyncioTestCase):
    setUp=chat_fixture.ContractIntegrationTests.setUp
    run_turn=chat_fixture.ContractIntegrationTests.run_turn

    async def test_request_override_reaches_answer_service_without_extra_router_call(self):
        self.payload.answer_guidance=False
        self.fixture.decision={'route':'university_test'}
        self.module.resolve_search_query=lambda q,*a:(q,[])
        await self.run_turn()
        self.assertFalse(self.llm.get_legal_response_stream.call_args.kwargs['answer_guidance'])
        self.assertEqual(len(self.fixture.requests),1)


class GuidanceGenerationTests(unittest.IsolatedAsyncioTestCase):
    setUp=fixture.DiagnosticTests.setUp
    answering_module=fixture.DiagnosticTests.answering_module

    async def test_on_off_use_separate_cache_entries_and_correct_payloads(self):
        answering=self.answering_module()
        answering.load_domain_config=lambda *a:self.config
        answering.config_for_mode=lambda c,m:c
        answering.mode_settings=lambda *a:{}
        answering.build_evidence_bundle=AsyncMock(return_value=(json.dumps({'Chủ đề':'Card','Nội dung':'2 days'}),.9,['Card']))
        events=[];answering.emit=lambda task,event='info',**kw:events.append((task,kw))
        payloads=[]
        class Response:
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            def raise_for_status(self):pass
            async def aiter_lines(self):
                yield 'data: '+json.dumps({'choices':[{'delta':{'content':'2 days'}}]})
                yield 'data: [DONE]'
        class Client:
            def __init__(self,**kw):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            def stream(self,*a,**kw):payloads.append(kw['json']);return Response()
        onboarding=types.ModuleType('rag.domain_onboarding');onboarding.GENERIC_EVIDENCE_PROMPT='Use evidence'
        fastapi=types.ModuleType('fastapi');fastapi.HTTPException=type('HTTPException',(Exception,),{})
        with patch.dict(sys.modules,{'rag.domain_onboarding':onboarding,'fastapi':fastapi}),patch.object(answering.httpx,'AsyncClient',Client):
            for setting in [True,False,False,True]:
                stream,_=await answering.get_legal_response_stream('time?','time?',answer_guidance=setting)
                self.assertIn('2 days',''.join([frame async for frame in stream]))
        self.assertEqual(len(payloads),2)
        self.assertEqual(answering.build_evidence_bundle.await_count,2)
        self.assertIn('Before answering, silently',payloads[0]['messages'][0]['content'])
        self.assertNotIn('Before answering, silently',payloads[1]['messages'][0]['content'])
        settings=[v['runtime_options']['RAG_ANSWER_GUIDANCE'] for t,v in events if t=='answer.cache' and not v['cache_hit']]
        self.assertEqual(settings,[True,False])


class TargetedFixtureTests(unittest.TestCase):
    def test_matched_variants_differ_only_in_guidance_and_order_is_balanced(self):
        manifest=json.loads((ROOT/'testing/fixtures/qa_targeted_ab_v1.json').read_text())
        groups={}
        for case in manifest['scenarios']:
            if case.get('warmup'):continue
            groups.setdefault(case['comparison_group'],[]).append(case)
        first_settings=[]
        for label,cases in groups.items():
            self.assertEqual(len(cases),2)
            first_settings.append(cases[0]['answer_guidance'])
            left,right=cases
            self.assertNotEqual(left['answer_guidance'],right['answer_guidance'])
            self.assertEqual(left.get('mode','strict'),right.get('mode','strict'))
            for a,b in zip(left['turns'],right['turns']):
                self.assertEqual(a['prompt'],b['prompt']);self.assertEqual(a['expected'],b['expected'])
                self.assertEqual({k:v for k,v in a.get('options',{}).items() if k!='answer_guidance'},
                                 {k:v for k,v in b.get('options',{}).items() if k!='answer_guidance'})
        self.assertLessEqual(abs(first_settings.count(True)-first_settings.count(False)),1)
        self.assertEqual(sum(len(c['turns']) for c in manifest['scenarios']),65)

    def test_analysis_excludes_warmup_and_preserves_answers(self):
        analysis=load('target_analysis_test','testing/targeted_analysis.py')
        cases=[{'id':'warm','warmup':True},
               {'id':'on','comparison_group':'g','answer_guidance':True},
               {'id':'off','comparison_group':'g','answer_guidance':False}]
        rows=[{'scenario':c['id'],'turn':1,'prompt':'question','status':'PASS','elapsed_ms':ms,
               'actual':{'stream':{'answer':'No extra caveat'}}} for c,ms in zip(cases,[100000,12000,11000])]
        result=analysis.analyze({'scenarios':cases},rows)
        self.assertEqual(result['on']['mean_seconds'],12)
        self.assertEqual(result['off']['mean_seconds'],11)
        self.assertEqual(len(result['pairs']),1)
        rows.pop()
        with self.assertRaises(ValueError):analysis.analyze({'scenarios':cases},rows)

    def test_targeted_driver_packages_paired_review_and_keeps_default_suites(self):
        import tempfile
        import shutil
        import zipfile
        import io
        driver=load('targeted_driver_test','testing/run_cacheless_tests.py')
        analysis=load('targeted_analysis','testing/targeted_analysis.py')
        manifest=json.loads((ROOT/'testing/fixtures/qa_targeted_ab_v1.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            temporary=Path(folder)
            (temporary/'testing/fixtures').mkdir(parents=True)
            for name in ('qa_targeted_ab_v1.json','qa_frozen_baseline_20261022.json'):
                shutil.copyfile(ROOT/'testing/fixtures'/name,temporary/'testing/fixtures'/name)
            def fake_run(command,**kwargs):
                self.assertIn('--cacheless',command)
                self.assertIn('--require-all-domains',command)
                results=[]
                for case in manifest['scenarios']:
                    for turn,step in enumerate(case['turns'],1):
                        results.append({'scenario':case['id'],'turn':turn,'repeat':1,
                            'prompt':step['prompt'],'status':'PASS','elapsed_ms':1000,
                            'actual':{'stream':{'answer':'fixture answer'}}})
                run={'turn_count':65,'counts':{'PASS':65},'answer_cache_bypassed':True,
                     'performance':{'answer_cache_events':50,'cache_hits':0,'cache_bypass_acknowledged':50}}
                directory=temporary/'qa_exports';directory.mkdir()
                with zipfile.ZipFile(directory/'system_tests_fake.zip','w') as bundle:
                    for name,value in [('results.json',results),('run.json',run),('qa_cases.json',manifest)]:
                        bundle.writestr(name,json.dumps(value))
            with patch.object(driver,'ROOT',temporary),patch.object(driver.subprocess,'run',side_effect=fake_run),                 patch.object(sys,'argv',['runner','--suite','targeted']),patch.dict(sys.modules,{'targeted_analysis':analysis}):
                driver.main()
            combined=list((temporary/'qa_exports').glob('cacheless_comparison_*.zip'))
            self.assertEqual(len(combined),1)
            with zipfile.ZipFile(combined[0]) as outer:
                with zipfile.ZipFile(io.BytesIO(outer.read('system_tests_fake.zip'))) as inner:
                    review=json.loads(inner.read('guidance_comparison.json'))
                    self.assertEqual(review['on']['turn_count'],32)
                    self.assertEqual(review['off']['turn_count'],32)
            self.assertEqual(driver.SUITES['original_64'],'qa_baseline_64_v1.json')
            self.assertEqual(driver.SUITES['fresh_48'],'qa_fresh_cases_v1.json')

if __name__=='__main__':unittest.main()
