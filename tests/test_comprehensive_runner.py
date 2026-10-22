import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
from testing import run_comprehensive_tests as runner


class ComprehensiveRunnerTests(unittest.TestCase):
    def source_db(self,folder):
        profile=json.loads((runner.FIXTURE/'xuong_nho_profile.json').read_text(encoding='utf-8'))
        rows=json.loads((runner.FIXTURE/'xuong_nho_dataset.json').read_text(encoding='utf-8'))
        keys=list(profile['schema_mapping'])
        path=Path(folder)/'records.db'
        with sqlite3.connect(path) as db:
            columns=', '.join('"'+k+'" TEXT' for k in keys)
            db.execute(f'CREATE TABLE records ({columns})')
            db.executemany('INSERT INTO records VALUES ('+','.join('?' for _ in keys)+')',
                [[row[profile['schema_mapping'][k]] for k in keys] for row in rows])
        return types.SimpleNamespace(profile=profile,schema=profile['schema_mapping'],db_path=path,
            table='records',name='maker_lab_test',generation='g',revision='r')

    def test_preflight_checks_already_ingested_rows_without_modifying_database(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.source_db(folder)
            before=config.db_path.read_bytes()
            with patch('rag.domain_config.load_domain_config',return_value=config):
                result=runner.preflight()
            self.assertEqual(result['fixture_rows'],12)
            self.assertEqual(config.db_path.read_bytes(),before)

    def test_mismatched_data_stops_instead_of_reimporting(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.source_db(folder)
            with sqlite3.connect(config.db_path) as db:
                db.execute('DELETE FROM records WHERE rowid=1')
            before=config.db_path.read_bytes()
            with patch('rag.domain_config.load_domain_config',return_value=config),self.assertRaises(RuntimeError):
                runner.preflight()
            self.assertEqual(config.db_path.read_bytes(),before)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.source_db(folder)
            config.db_path=Path(folder)/'absent.db'
            with patch('rag.domain_config.load_domain_config',return_value=config),self.assertRaises(sqlite3.Error):
                runner.preflight()
            self.assertFalse(config.db_path.exists())

    def test_one_runner_preserves_every_test_suite_and_archives_fixture_metadata(self):
        real_root=runner.ROOT
        for suite,turns in [('all',64),('original',41),('novel',23),('smoke',41),('review',41)]:
            with self.subTest(suite=suite),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                for name in ('run_services.py','llm_service/main.py','rag/service_config.py'):
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(real_root/name,target)
                def fake_qa():
                    cases=Path(sys.argv[sys.argv.index('--cases')+1])
                    manifest=json.loads(cases.read_text(encoding='utf-8'))
                    self.assertEqual(sum(len(c['turns']) for c in manifest['scenarios']),turns)
                    self.assertIn('--require-all-domains',sys.argv)
                    if suite in {'smoke','review'}:
                        self.assertEqual(sys.argv[sys.argv.index('--suite')+1],suite)
                    directory=root/'qa_exports';directory.mkdir()
                    with zipfile.ZipFile(directory/'system_tests_fake.zip','w') as z:
                        z.writestr('run.json','{}')
                with (patch.object(runner,'ROOT',root),patch.object(runner,'preflight',return_value={'fixture_rows':12}) as check,
                     patch.object(runner.qa,'main',side_effect=fake_qa),patch('sys.argv',['runner','--suite',suite])):
                    runner.main()
                self.assertEqual(check.call_count,1 if suite in {'all','novel'} else 0)
                with zipfile.ZipFile(root/'qa_exports/system_tests_fake.zip') as z:
                    info=json.loads(z.read('testing_run.json'))
                    self.assertEqual(info['suite'],suite)
                    self.assertEqual('novel_preflight.json' in z.namelist(),suite in {'all','novel'})


if __name__=='__main__':unittest.main()
