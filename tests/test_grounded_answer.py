"""Evidence contract tests with real public fixture values, without a live model."""
import json
from pathlib import Path
import types
import unittest
from rag.grounded_answer import public_documents, render_plan, normalize_plan_text, MISSING


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        profile=json.loads((root/'testing/fixtures/xuong_nho/xuong_nho_profile.json').read_text(encoding='utf-8'))
        self.config=types.SimpleNamespace(primary_entity=profile['primary_entity'],
            schema=profile['schema_mapping'],evidence_fields=[k for k in profile['schema_mapping']
                if k not in profile.get('excluded_fields',[])])
        self.rows=json.loads((root/'testing/fixtures/xuong_nho/xuong_nho_dataset.json').read_text(encoding='utf-8'))
        self.title=self.config.schema[self.config.primary_entity]

    def doc(self,title):
        return next(row for row in self.rows if row[self.title]==title)

    def render(self,doc,fields):
        return render_plan(json.dumps([{'record':0,'fields':fields}]),[doc],self.config)

    def test_room_limits_use_exact_public_values_without_organization_caveat(self):
        for station in ('Trạm Mây','Trạm Nắng'):
            doc=self.doc('Đặt phòng làm việc — '+station)
            text,fields=self.render(doc,[self.config.schema['policy_details']])
            self.assertIn(str(doc[self.config.schema['policy_details']]),text)
            self.assertNotIn('không',text.lower())
            self.assertIn('policy_details',fields)

    def test_full_record_recovers_fields_lost_by_first_field_vote(self):
        doc=self.doc('Đặt phòng làm việc — Trạm Nắng')
        doc={**doc,'_raw_json':'SECRET','private_email':'SECRET'}
        context=json.dumps({self.title:doc[self.title],self.config.schema['policy_details']:doc[self.config.schema['policy_details']]})
        full=public_documents(context,['RAG_CONTENT:'+json.dumps(doc)],self.config)
        self.assertNotIn('SECRET',json.dumps(full))
        labels=[self.config.schema[k] for k in ('policy_details','service_charge','service_desk')]
        text,_=self.render(full[0],labels)
        self.assertIn('45000',text)
        self.assertIn('Quầy không gian Trạm Nắng',text)

    def test_repair_preserves_time_price_and_units(self):
        doc=self.doc('Sửa máy khoan — Trạm Mây')
        labels=[self.config.schema[k] for k in ('request_turnaround','service_charge')]
        text,_=self.render(doc,labels)
        for label in labels:self.assertIn(str(doc[label]),text)
        self.assertNotIn('Không có thông tin',text)

    def test_missing_deposit_does_not_substitute_service_charge(self):
        doc=self.doc('Mượn máy chiếu — Trạm Mây')
        text,_=self.render(doc,[])
        self.assertIn(MISSING,text)
        self.assertNotIn(str(doc[self.config.schema['service_charge']]),text)

    def test_invalid_plans_cannot_add_prose_or_fake_values(self):
        doc=self.doc('Mượn máy khoan — Trạm Mây')
        cases=[
            'Correct answer. No information about Xưởng Nhỏ.',
            '[{"record":0,"fields":["invented_field"]}]',
            '[{"record":0,"fields":[],"caveat":"unsupported"}]',
            '[{"record":true,"fields":[]}]',
            '[{"record":-1,"fields":[]}]',
            '[{"record":99,"fields":[]}]',
            '[]',
        ]
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                render_plan(raw,[doc],self.config)

    def test_only_complete_outer_array_closure_can_be_normalized(self):
        for raw in ('[{"record":0,"fields":["Quy tắc và giới hạn"]',
                    '[{"record":0,"fields":["Quy tắc',
                    '[{"record":0,"fields":[],',
                    '[{"record":0,"fields":[]} extra prose',
                    '[{"record":0,"fields":[]},',
                    '[{"record":0,"fields":[]} ] extra prose'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):normalize_plan_text(raw)
        valid='[{"record":0,"fields":[]}]'
        self.assertEqual(normalize_plan_text(valid),(valid,None))
        self.assertEqual(normalize_plan_text(valid[:-1]),(valid,'closed_outer_array'))

    def test_normalization_cannot_bypass_record_or_field_validation(self):
        doc=self.doc('Mượn máy khoan — Trạm Mây')
        for raw in ('[{"record":99,"fields":[]}',
                    '[{"record":true,"fields":[]}',
                    '[{"record":0,"fields":["invented"]}',
                    '[{"record":0,"fields":[],"answer":"invented"}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):render_plan(raw,[doc],self.config)

    def test_cross_record_mixup_and_omissions_are_rejected(self):
        docs=[self.doc('Mượn máy khoan — Trạm Mây'),self.doc('Mượn máy khoan — Trạm Nắng')]
        for plan in [[{'record':0,'fields':[]}], [{'record':0,'fields':[]},{'record':0,'fields':[]}]]:
            with self.assertRaises(ValueError):render_plan(json.dumps(plan),docs,self.config)
        label=self.config.schema['policy_details']
        text,_=render_plan(json.dumps([{'record':1,'fields':[label]},{'record':0,'fields':[label]}]),docs,self.config)
        self.assertLess(text.index('Trạm Mây'),text.index('Trạm Nắng'))
        for doc in docs:self.assertIn(doc[label],text)

    def test_zero_values_and_unicode_are_preserved(self):
        label=self.config.schema['service_charge']
        doc={self.title:'Ví dụ',label:0}
        self.assertIn(': 0',self.render(doc,[label])[0])
        self.assertIn(MISSING,self.render({self.title:'Ví dụ',label:None},[label])[0])


if __name__=='__main__':unittest.main()
