import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from rag.service_config import llm_port,llm_url

ROOT=Path(__file__).resolve().parents[1]


class ServiceConfigTests(unittest.TestCase):
    def test_default_server_and_client_share_8001(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(llm_port(),8001)
            self.assertEqual(llm_url(),'http://127.0.0.1:8001')

    def test_port_override_moves_default_client_with_server(self):
        with patch.dict(os.environ,{'LLM_PORT':'8101'},clear=True):
            self.assertEqual(llm_port(),8101)
            self.assertEqual(llm_url(),'http://127.0.0.1:8101')

    def test_existing_url_overrides_retain_precedence(self):
        with patch.dict(os.environ,{'LLM_SERVICE_URL':'http://example.test:9001/',
            'LLM_SERVICE_URLS':' http://localhost:8001/,http://localhost:8101 '},clear=True):
            self.assertEqual(llm_url(),'http://localhost:8001')

    def test_invalid_local_port_is_rejected(self):
        for value in ('0','65536','abc'):
            with self.subTest(value=value),patch.dict(os.environ,{'LLM_PORT':value},clear=True):
                with self.assertRaises(ValueError):llm_port()

    def test_launcher_uses_matching_readiness_port_and_preserves_other_services(self):
        dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda *a:None
        spec=importlib.util.spec_from_file_location('test_service_launcher',ROOT/'run_services.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'dotenv':dotenv}):spec.loader.exec_module(module)
        for port in (8001,8101):
            with patch.dict(os.environ,{'LLM_PORT':str(port)},clear=True):
                services=module.service_definitions()
                self.assertEqual([s[3] for s in services],[port,8000,8501])
                self.assertEqual(services[0][4],'/v1/rag/status')
        tree=ast.parse((ROOT/'llm_service/main.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='uvicorn' and n.func.attr=='run']
        self.assertEqual(len(calls),1)
        port_arg=next(k.value for k in calls[0].keywords if k.arg=='port')
        self.assertIsInstance(port_arg,ast.Call)
        self.assertEqual(port_arg.func.id,'llm_port')


if __name__=='__main__':unittest.main()
