"""Test the context experiment using real temporary SQL and mocked model transport."""
import json
import os
import unittest
from unittest.mock import AsyncMock,patch
import test_pipeline_diagnostics as fixtures


class FullContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fixtures.DiagnosticTests.setUp(self)

    async def test_full_records_skip_field_vote_without_leaking_excluded_columns(self):
        self.config.schema={'topic':'Chủ đề','details':'Nội dung','fee':'Phí','description':'Private'}
        self.config.evidence_fields=['topic','details','fee']
        self.pipeline.retrieve_and_rerank=lambda *a,**kw:[(1,.9,{'topic':'Library'})]
        self.pipeline.classify_requested_columns=AsyncMock(side_effect=AssertionError('Column selector called'))
        context,_,sources=await self.pipeline.build_evidence_bundle('Fee?', 'Library. Fee?',
            config=self.config,active_procedures=['Library'],context_mode='full')
        doc=json.loads(context)
        self.assertEqual(doc,{'Chủ đề':'Library','Nội dung':'3 books, 14 days','Phí':'0'})
        self.assertNotIn('Actual content',context)
        self.assertNotIn('Private',context)
        self.pipeline.classify_requested_columns.assert_not_called()
        self.assertIn('Library',sources)
        self.assertTrue(any(e=='field_selection_skipped' for _,e,_ in self.events))

    async def test_default_projection_still_uses_field_selection(self):
        self.config.schema={'topic':'Chủ đề','details':'Nội dung','fee':'Phí'}
        self.config.evidence_fields=list(self.config.schema)
        self.pipeline.retrieve_and_rerank=lambda *a,**kw:[(1,.9,{'topic':'Library'})]
        self.pipeline.classify_requested_columns=AsyncMock(return_value=['fee'])
        context,_,_=await self.pipeline.build_evidence_bundle('Fee?', 'Library. Fee?',
            config=self.config,active_procedures=['Library'])
        self.assertEqual(json.loads(context),{'Chủ đề':'Library','Phí':'0'})
        self.pipeline.classify_requested_columns.assert_awaited_once()

    async def test_unknown_context_mode_fails_before_retrieval(self):
        self.pipeline.retrieve_and_rerank=lambda *a,**kw:self.fail('Retrieval called')
        with self.assertRaises(ValueError):
            await self.pipeline.build_evidence_bundle('Fee?','Fee?',config=self.config,context_mode='invalid')

    def test_full_context_is_default_and_selected_mode_remains_available(self):
        answering=fixtures.DiagnosticTests.answering_module(self)
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(answering.evidence_context_mode(),'full')
        with patch.dict(os.environ,{'RAG_CONTEXT_MODE':'selected'}):
            self.assertEqual(answering.evidence_context_mode(),'selected')
        with patch.dict(os.environ,{'RAG_CONTEXT_MODE':'full'}):
            self.assertEqual(answering.evidence_context_mode(),'full')
        with patch.dict(os.environ,{'RAG_CONTEXT_MODE':'invalid'}):
            with self.assertRaises(ValueError):answering.evidence_context_mode()


if __name__=='__main__':unittest.main()
