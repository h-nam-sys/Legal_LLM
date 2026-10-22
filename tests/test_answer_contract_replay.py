"""Replay exact raw outputs captured from the user's live model."""
import json
from pathlib import Path
import types
import unittest
from rag.grounded_answer import normalize_plan_text, render_plan


class CapturedAnswerTests(unittest.TestCase):
    def test_captured_outputs_preserve_source_facts_and_admissions_control(self):
        cases=json.loads((Path(__file__).parent/'fixtures/answer_contract_replay.json').read_text(encoding='utf-8'))
        self.assertEqual(len(cases),3)
        for case in cases:
            with self.subTest(case=case['scenario']):
                config=types.SimpleNamespace(primary_entity=case['primary_entity'],schema=case['schema'],
                    evidence_fields=case['evidence_fields'])
                normalized,repair=normalize_plan_text(case['raw_model_text'])
                self.assertEqual(repair,case['expected_normalization'])
                text,fields=render_plan(case['raw_model_text'],case['documents'],config)
                for fact in case['required_text']:
                    self.assertIn(fact,text)
                if case.get('expected_rendered_answer'):
                    self.assertEqual(text,case['expected_rendered_answer'])
                self.assertEqual(fields,case['expected_fields'])
                if repair:
                    self.assertEqual(normalized,case['raw_model_text']+']')
                self.assertNotIn('Lỗi hệ thống',text)


if __name__=='__main__':unittest.main()
