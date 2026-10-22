"""Reference decisions are mocked; parsing and question construction are real."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests=[];self.raw='[0]';owner=self
        httpx=types.ModuleType('httpx')
        httpx.HTTPError=type('HTTPError',(Exception,),{})
        class Client:
            def __init__(self,**kw):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            async def post(self,url,**kw):
                owner.requests.append(kw['json'])
                return types.SimpleNamespace(raise_for_status=lambda:None,
                    json=lambda:{'choices':[{'message':{'content':owner.raw}}]})
        httpx.AsyncClient=Client
        spec=importlib.util.spec_from_file_location('rag.turn_context_under_test',ROOT/'rag/turn_context.py')
        self.module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'httpx':httpx}):spec.loader.exec_module(self.module)
        self.module.emit=lambda *a,**kw:None
        self.config=types.SimpleNamespace(name='test',profile={'description':'Equipment workshop and branches'})

    async def test_action_continuation_keeps_identity_without_rewriting_question(self):
        question='Gia hạn được mấy lần, thêm bao nhiêu ngày?'
        resolved=await self.module.resolve_references(question,['Mượn máy khoan — Trạm Mây'],self.config)
        self.assertIn(question,resolved)
        self.assertIn('Trạm Mây',resolved)
        self.assertIn('current action',resolved)
        request=json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(request['current_question'],question)
        self.assertNotIn('route',request)

    async def test_new_subject_is_returned_verbatim_without_old_identity(self):
        self.raw='[]'
        question='Xin bảng điểm tại Demo có phí không?'
        self.assertEqual(await self.module.resolve_references(question,['Cấp lại thẻ sinh viên'],self.config),question)

    async def test_no_subject_skips_inference(self):
        self.assertEqual(await self.module.resolve_references('New topic',[],self.config),'New topic')
        self.assertFalse(self.requests)

    async def test_invalid_indices_fail_closed(self):
        for raw in ('[true]','[-1]','[2]','["0"]','{"route":"followup"}','not JSON'):
            self.raw=raw
            with self.subTest(raw=raw),self.assertRaises(self.module.RAGUnavailableError):
                await self.module.resolve_references('And renewal?',['Old subject'],self.config)

    async def test_multiple_references_are_deduplicated(self):
        self.raw='[1,0,1]'
        resolved=await self.module.resolve_references('Compare both',['Branch A','Branch B'],self.config)
        self.assertEqual(resolved.count('Branch B'),1)
        self.assertEqual(resolved.count('Branch A'),1)


if __name__=='__main__':unittest.main()
