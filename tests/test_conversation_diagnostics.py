"""No model, database, GPU, or web server required."""
import asyncio
import json
from pathlib import Path
import queue
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import conversation_diagnostics as diag
from rag import timing

class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.queue = queue.Queue()
        worker = types.SimpleNamespace(is_alive=lambda: True)
        self.patches = [patch.object(diag, '_queue', self.queue), patch.object(diag, '_thread', worker)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def entries(self):
        records = []
        while not self.queue.empty():
            records.append(self.queue.get_nowait())
        # Verify actual JSON roundtrip, including accents and embedded newlines.
        return [json.loads(json.dumps(r, ensure_ascii=False)) for r in records]

    def payload(self, text='Đăng ký khai sinh?'):
        return types.SimpleNamespace(user_prompt=text, domain='auto', mode='strict')

    async def consume(self, response):
        return [chunk async for chunk in response.body_iterator]

    async def test_saved_response_and_routing_with_unchanged_stream(self):
        frames = ['data: {"type":"chunk","text":"Chào bạn\\n"}\n\n', 'data: [DONE]\n\n']
        @diag.capture_turn
        async def chat(cid, payload, db):
            diag.routing_event('raw_decision', content='{"kind":"domain","domain":"demo"}')
            diag.routing_event('decision', decision={'kind': 'domain', 'domain': 'demo'})
            diag.routing_event('selected', domain='demo')
            diag.record('domain_state', state={'name': 'demo', 'selection': 'auto'})
            async def stream():
                yield frames[0]
                diag.record('message', conversation_id=cid, role='assistant', content='Chào bạn\n', message_id=42, sources=['RAG_DOMAIN:demo'])
                diag.record('interaction', outcome='SUCCESS', rag_score=0.9)
                yield frames[1]
            return types.SimpleNamespace(body_iterator=stream(), headers={'unchanged': True})
        token = timing._trace.set('a' * 16)
        try:
            response = await chat(7, self.payload(), None)
        finally:
            timing._trace.reset(token)
        self.assertEqual(await self.consume(response), frames)
        self.assertTrue(response.headers['unchanged'])
        turn = next(r for r in self.entries() if r['event'] == 'turn')
        self.assertEqual(turn['user_prompt'], 'Đăng ký khai sinh?')
        self.assertEqual(turn['assistant_response'], 'Chào bạn\n')
        self.assertEqual(turn['request_id'], 'a' * 16)
        self.assertEqual(turn['selected_domain'], 'demo')
        self.assertEqual(turn['classifier_decision']['kind'], 'domain')
        self.assertEqual(turn['status'], 'ok')
        self.assertTrue(turn['response_saved'])
        self.assertGreaterEqual(turn['elapsed_ms'], 0)
        self.assertIsNone(diag._turn.get())

    async def test_error_before_stream_is_logged_and_reraised(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            raise ValueError('retrieval failed')
        with self.assertRaisesRegex(ValueError, 'retrieval failed'):
            await chat(3, self.payload(), None)
        turn = self.entries()[-1]
        self.assertEqual(turn['status'], 'error')
        self.assertEqual(turn['error_type'], 'ValueError')
        self.assertFalse(turn['response_saved'])
        self.assertIsNone(diag._turn.get())

    async def test_stream_error_keeps_partial_text(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            async def stream():
                yield 'data: {"type":"chunk","text":"partial"}\n\n'
                raise RuntimeError('model disconnected')
            return types.SimpleNamespace(body_iterator=stream())
        response = await chat(2, self.payload(), None)
        with self.assertRaisesRegex(RuntimeError, 'model disconnected'):
            await self.consume(response)
        turn = self.entries()[-1]
        self.assertEqual(turn['assistant_response'], 'partial')
        self.assertFalse(turn['response_saved'])
        self.assertEqual(turn['status'], 'error')

    async def test_cancelled_stream_and_cleanup(self):
        closed = []
        @diag.capture_turn
        async def chat(cid, payload, db):
            async def stream():
                try:
                    yield 'data: {"type":"chunk","text":"partial"}\n\n'
                    yield 'data: [DONE]\n\n'
                finally:
                    closed.append(True)
            return types.SimpleNamespace(body_iterator=stream())
        response = await chat(2, self.payload(), None)
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        self.assertTrue(closed)
        self.assertEqual(self.entries()[-1]['status'], 'cancelled')
        self.assertIsNone(diag._turn.get())

    async def test_concurrent_turns_remain_isolated(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            diag.record('domain_state', state={'name': 'domain' + str(cid), 'selection': 'manual'})
            async def stream():
                await asyncio.sleep(0)
                diag.record('message', role='assistant', content='answer' + str(cid), sources=[])
                yield 'data: [DONE]\n\n'
            return types.SimpleNamespace(body_iterator=stream())
        async def run(cid):
            return await self.consume(await chat(cid, self.payload(str(cid)), None))
        await asyncio.gather(run(1), run(2))
        turns = {r['conversation_id']: r for r in self.entries() if r['event'] == 'turn'}
        for cid in (1, 2):
            self.assertEqual(turns[cid]['assistant_response'], 'answer' + str(cid))
            self.assertEqual(turns[cid]['selected_domain'], 'domain' + str(cid))
            self.assertNotIn('classifier_decision', turns[cid])

    async def test_clarification_summary(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            diag.routing_event('clarification', reason='decision_validation_failed')
            async def stream():
                diag.record('message', role='assistant', content='Chọn lĩnh vực?', sources=[])
                diag.record('interaction', outcome='DOMAIN_CLARIFICATION')
                yield 'data: [DONE]\n\n'
            return types.SimpleNamespace(body_iterator=stream())
        await self.consume(await chat(1, self.payload(), None))
        turn = self.entries()[-1]
        self.assertEqual(turn['clarification_reason'], 'decision_validation_failed')
        self.assertEqual(turn['outcome'], 'DOMAIN_CLARIFICATION')
        self.assertIsNone(turn['selected_domain'])

    async def test_queue_full_does_not_break_answer(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            async def stream():
                diag.record('message', role='assistant', content='answer')
                yield 'data: [DONE]\n\n'
            return types.SimpleNamespace(body_iterator=stream())
        with patch.object(self.queue, 'put_nowait', side_effect=queue.Full), patch.object(diag, '_warning') as warning:
            self.assertEqual(await self.consume(await chat(1, self.payload(), None)), ['data: [DONE]\n\n'])
            self.assertTrue(warning.called)

    async def test_missing_saved_answer_is_incomplete(self):
        @diag.capture_turn
        async def chat(cid, payload, db):
            async def stream():
                yield 'data: [DONE]\n\n'
            return types.SimpleNamespace(body_iterator=stream())
        await self.consume(await chat(1, self.payload(), None))
        self.assertEqual(self.entries()[-1]['status'], 'incomplete')


class ExportTests(unittest.TestCase):
    def test_export_collects_project_logs_and_readable_transcript(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('collect_diagnostics.py'))
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        self.assertEqual(collector.ROOT, Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'logs/conversations/test.jsonl'
            log.parent.mkdir(parents=True)
            record = dict(event='turn', user_prompt='Phí bao nhiêu?\nChi tiết?', assistant_response='8.000đ',
                          request_id='a' * 16, conversation_id=4, status='ok', elapsed_ms=1234,
                          selected_domain='demo', classifier_decision={'kind': 'domain', 'domain': 'demo'})
            log.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
            session, files = collector.recent_files(root)
            self.assertIn(log, files)
            text = collector.transcript(files)
            self.assertIn(record['user_prompt'], text)
            self.assertIn(record['assistant_response'], text)
            self.assertIn('1234', text)
            self.assertEqual(collector.inspect(root, files)['turn_count'], 1)


if __name__ == '__main__':
    unittest.main()

class WriterTests(unittest.TestCase):
    def test_writer_produces_utf8_jsonl_and_drains_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = queue.Queue()
            items.put({'event': 'turn', 'user_prompt': 'Phí?\nBao nhiêu?', 'assistant_response': '8.000đ'})
            items.put(None)
            with patch.object(diag, '_queue', items), patch.object(diag, '__file__', str(root / 'rag/conversation_diagnostics.py')):
                diag._write()
            paths = list((root / 'logs/conversations').glob('*.jsonl'))
            self.assertEqual(len(paths), 1)
            lines = paths[0].read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])['user_prompt'], 'Phí?\nBao nhiêu?')
            self.assertEqual(items.unfinished_tasks, 0)

    def test_unwritable_log_path_only_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').write_text('blocks directory creation')
            with patch.object(diag, '__file__', str(root / 'rag/conversation_diagnostics.py')), patch.object(diag, '_warning') as warning:
                diag._write()
            warning.assert_called_once()
