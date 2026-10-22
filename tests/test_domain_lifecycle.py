"""Run with: python -m unittest discover -s tests -p test_domain_lifecycle.py"""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag import domain_config as dc
from rag import domain_management as management
from rag import ingestion_jobs as jobs


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('legal', 'campus', 'student'):
            self.make_domain(name)
        dc.atomic_json(self.root / 'data/domain_profile.json', self.profile('legal'))
        for module in (dc, jobs, management):
            mock = patch.object(module, 'ROOT_DIR', self.root)
            mock.start()
            self.addCleanup(mock.stop)

    def profile(self, name):
        return {'domain_name': name, 'primary_entity': 'topic',
                'schema_mapping': {'topic': 'Topic', 'details': 'Details'}}

    def make_domain(self, name):
        profile = self.profile(name)
        dc.atomic_json(self.root / 'data/domains' / f'{name}.json', profile)
        directory = self.root / 'data/rag_generations' / name / ('a' * 32)
        dc.atomic_json(directory / 'profile.json', profile)
        (directory / 'records.db').write_bytes(b'original database')
        (directory / 'vectors').mkdir()
        dc.atomic_json(directory / 'vectors/docstore.json', {'original': 'vectors'})
        dc.atomic_json(directory.parent / 'current.json', {'generation': 'a' * 32})

    def test_unchanged_defaults(self):
        self.assertEqual(dc.load_domain_config().name, 'legal')
        self.assertEqual({c.name for c in dc.list_domain_configs()}, {'legal', 'campus', 'student'})

    def test_disable_enable_and_data_preservation(self):
        before = dc.load_domain_config('campus')
        management.set_domain_enabled('campus', False)
        self.assertNotIn('campus', {c.name for c in dc.list_domain_configs()})
        self.assertIn('campus', {c.name for c in dc.list_domain_configs(include_inactive=True)})
        with self.assertRaises(dc.RAGConfigurationError):
            dc.load_domain_config('campus')
        self.assertEqual(dc.load_domain_config('campus', allow_inactive=True).revision, before.revision)
        self.assertEqual(before.db_path.read_bytes(), b'original database')
        management.set_domain_enabled('campus', True)
        self.assertEqual(dc.load_domain_config('campus').revision, before.revision)

    def test_settings_reset_does_not_enable(self):
        management.set_domain_enabled('campus', False)
        settings = self.root / 'data/domain_settings/campus.json'
        dc.atomic_json(settings, {'description': 'updated'})
        settings.unlink()
        self.assertEqual(dc.domain_status('campus'), 'disabled')

    def test_default_protected(self):
        for action in (lambda: management.set_domain_enabled('legal', False),
                       lambda: management.delete_domain('legal', 'legal')):
            with self.assertRaises(dc.RAGConfigurationError):
                action()
        self.assertEqual(dc.load_domain_config().name, 'legal')

    def test_exact_confirmation(self):
        with self.assertRaises(dc.RAGConfigurationError):
            management.delete_domain('campus', 'wrong')
        self.assertEqual(dc.domain_status('campus'), 'enabled')

    def test_delete_archives_only_owned_data(self):
        dc.atomic_json(self.root / 'data/domain_settings/campus.json', {'default_mode': 'friendly'})
        upload = self.root / 'data/admin_uploads/shared.csv'
        upload.parent.mkdir(parents=True)
        upload.write_text('shared')
        original = dc.load_domain_config('campus')
        archive = management.delete_domain('campus', 'campus')
        self.assertFalse(original.db_path.exists())
        self.assertEqual((archive / original.db_path.relative_to(self.root)).read_bytes(), b'original database')
        self.assertEqual(dc.domain_status('campus'), 'deleted')
        self.assertNotIn('campus', {c.name for c in dc.list_domain_configs(include_inactive=True)})
        self.assertTrue(upload.exists())
        self.assertTrue(dc.load_domain_config('legal').db_path.exists())
        self.assertTrue(dc.load_domain_config('student').db_path.exists())
        with self.assertRaises(dc.RAGConfigurationError):
            dc.load_domain_config('campus', allow_inactive=True)
        with self.assertRaises(dc.RAGConfigurationError):
            management.set_domain_enabled('campus', True)

    def test_failed_move_rolls_back_enabled_and_disabled(self):
        for enabled in (True, False):
            management.set_domain_enabled('campus', enabled)
            rename = management.os.rename
            calls = []
            def fail_second(source, destination):
                calls.append(source)
                if len(calls) == 2:
                    raise PermissionError('open Windows DB')
                return rename(source, destination)
            with patch.object(management.os, 'rename', side_effect=fail_second):
                with self.assertRaises(dc.RAGConfigurationError):
                    management.delete_domain('campus', 'campus')
            self.assertEqual(dc.domain_status('campus'), 'enabled' if enabled else 'disabled')
            self.assertTrue(dc.load_domain_config('campus', allow_inactive=True).db_path.exists())
            self.assertTrue((self.root / 'data/domains/campus.json').exists())

    def test_active_ingestion_blocks_toggle_and_delete(self):
        lock = jobs.claim('campus', 'job1', self.root)
        try:
            for action in (lambda: management.set_domain_enabled('campus', False),
                           lambda: management.delete_domain('campus', 'campus')):
                with self.assertRaises(dc.RAGConfigurationError):
                    action()
            self.assertTrue(dc.load_domain_config('campus').db_path.exists())
        finally:
            jobs.release(lock, 'job1')

    def test_deleted_domain_blocks_new_writers(self):
        management.delete_domain('campus', 'campus')
        with self.assertRaises(dc.RAGConfigurationError):
            jobs.claim('campus', 'new-job', self.root)
        self.assertFalse(jobs.lock_path('campus', self.root).exists())

    def test_path_traversal_rejected(self):
        with self.assertRaises(dc.RAGConfigurationError):
            management.delete_domain('../legal', '../legal')
        with self.assertRaises(dc.RAGConfigurationError):
            dc.domain_status('../legal')

    def test_shared_legacy_storage_not_deleted(self):
        name = 'legacy'
        p = self.profile(name)
        shared = self.root / 'data/shared.db'
        shared.write_text('do not delete')
        p['storage'] = {'database': 'data/shared.db', 'vectors': 'data/shared_vectors'}
        dc.atomic_json(self.root / 'data/domains/legacy.json', p)
        with self.assertRaises(dc.RAGConfigurationError):
            management.delete_domain(name, name)
        self.assertEqual(shared.read_text(), 'do not delete')

    def test_symlink_storage_rejected(self):
        owned = self.root / 'data/domains/campus'
        other = self.root / 'other'
        other.mkdir()
        try:
            owned.symlink_to(other, target_is_directory=True)
        except OSError:
            self.skipTest('Symlinks not available')
        with self.assertRaises(dc.RAGConfigurationError):
            management.delete_domain('campus', 'campus')
        self.assertTrue(other.exists())

    def load_router(self):
        # Keep the tests dependency-free. Only network/field helper imports are stubbed.
        httpx = types.ModuleType('httpx')
        fastapi = types.ModuleType('fastapi')
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                super().__init__(detail)
        fastapi.HTTPException = HTTPException
        qp = types.ModuleType('rag.query_pipeline')
        qp._plain = lambda text: text.lower()
        qp.explicit_fields = lambda *args: []
        with patch.dict(sys.modules, {'httpx': httpx, 'fastapi': fastapi, 'rag.query_pipeline': qp}):
            spec = importlib.util.spec_from_file_location('lifecycle_router', ROOT / 'backend/services/domain_router.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_router_excludes_disabled_and_deleted(self):
        routing = self.load_router()
        management.set_domain_enabled('campus', False)
        management.delete_domain('student', 'student')
        self.assertEqual([c.name for c in routing.ready_domains()], ['legal'])
        for name in ('campus', 'student'):
            with self.assertRaises(Exception) as raised:
                asyncio.run(routing.choose_domain('hi', {}, [], name))
            self.assertEqual(raised.exception.status_code, 400)

    def test_stale_saved_domain_switches_and_clears_titles(self):
        routing = self.load_router()
        management.set_domain_enabled('campus', False)
        seen = []
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, **kwargs):
                seen.append(kwargs['json']['messages'])
                return types.SimpleNamespace(raise_for_status=lambda: None,
                    json=lambda: {'choices': [{'message': {'content':
                        json.dumps({'route': 'student'})}}]})
        with patch.object(routing.httpx, 'AsyncClient', Client, create=True):
            config, state, changed = asyncio.run(routing.choose_domain(
                'new question', {'name': 'campus', 'selection': 'manual'}, ['Old campus title']))
        request = json.loads(seen[0][1]['content'])
        self.assertNotIn('followup_context_available', request)
        self.assertEqual(config.name, 'student')
        self.assertEqual(state['selection'], 'auto')
        self.assertTrue(changed)

    def test_no_ready_domain_returns_503(self):
        routing = self.load_router()
        with patch.object(routing, 'ready_domains', return_value=[]):
            with self.assertRaises(Exception) as raised:
                asyncio.run(routing.choose_domain('hi', {}, []))
            self.assertEqual(raised.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
