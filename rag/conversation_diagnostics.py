"""Write completed database messages off the request thread; never log tokens."""
import atexit
import contextvars
import functools
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from datetime import datetime, timezone

from .timing import trace_id

_queue = queue.Queue(maxsize=1024)
_guard = threading.Lock()
_thread = None
_warned = False
_turn = contextvars.ContextVar('conversation_diagnostic_turn', default=None)


def _capture(event, fields):
    turn = _turn.get()
    if turn is None:
        return
    if event == 'message' and fields.get('role') == 'assistant':
        turn['assistant_response'] = fields['content']
        turn['response_saved'] = True
        turn['message_id'] = fields.get('message_id')
        turn['sources'] = fields.get('sources', [])
    elif event == 'interaction':
        turn['outcome'] = fields.get('outcome')
        turn['rag_score'] = fields.get('rag_score')
    elif event == 'domain_state':
        turn['domain_state'] = dict(fields.get('state') or {})
    elif event == 'routing':
        step = fields.get('step')
        if step == 'decision':
            turn['classifier_decision'] = fields.get('decision')
        elif step == 'raw_decision':
            turn['classifier_raw_decision'] = fields.get('content')
        elif step == 'selected':
            turn['selected_domain'] = fields.get('domain')
        elif step == 'clarification':
            turn['clarification_reason'] = fields.get('reason')


def routing_event(step, **fields):
    record('routing', step=step, **fields)


def capture_turn(fn):
    """Observe the original response iterator; never generate or consume extra text."""
    @functools.wraps(fn)
    async def wrapped(conversation_id, payload, db):
        started = time.perf_counter()
        turn = dict(conversation_id=conversation_id, user_prompt=payload.user_prompt,
                    request_id=trace_id(), started_at=datetime.now(timezone.utc).isoformat(),
                    assistant_response=None, response_saved=False,
                    requested_domain=getattr(payload, 'domain', None),
                    requested_mode=getattr(payload, 'mode', None))

        def finish(status, exc=None):
            fields = {key: value for key, value in turn.items() if key != 'streamed_parts'}
            if not fields['response_saved'] and turn.get('streamed_parts'):
                fields['assistant_response'] = ''.join(turn['streamed_parts'])
            state = fields.get('domain_state', {})
            fields.setdefault('selected_domain', state.get('name'))
            fields.update(status=status, elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
            if exc is not None:
                fields.update(error_type=type(exc).__name__, error=str(exc))
            record('turn', **fields)

        token = _turn.set(turn)
        try:
            response = await fn(conversation_id, payload, db)
        except BaseException as exc:
            finish('cancelled' if isinstance(exc, (GeneratorExit,)) or type(exc).__name__ == 'CancelledError' else 'error', exc)
            raise
        finally:
            _turn.reset(token)
        source = getattr(response, 'body_iterator', None)
        if source is None:
            finish('ok')
            return response

        async def observed():
            status, failure = 'ok', None
            try:
                while True:
                    token = _turn.set(turn)
                    try:
                        value = await source.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _turn.reset(token)
                    # Existing chat generators emit complete SSE frames. Capture partial
                    # text for failed/disconnected turns, without changing those frames.
                    try:
                        text = value.decode('utf-8') if isinstance(value, bytes) else value
                        for line in text.splitlines():
                            if line.startswith('data:'):
                                data = json.loads(line[5:].strip())
                                if isinstance(data, dict) and data.get('type') == 'chunk':
                                    turn.setdefault('streamed_parts', []).append(data.get('text', ''))
                    except (ValueError, TypeError, AttributeError):
                        pass
                    yield value
            except BaseException as exc:
                status = 'cancelled' if isinstance(exc, GeneratorExit) or type(exc).__name__ == 'CancelledError' else 'error'
                failure = exc
                raise
            finally:
                try:
                    close = getattr(source, 'aclose', None)
                    if close is not None:
                        await close()
                finally:
                    if status == 'ok' and (not turn['response_saved'] or turn.get('outcome') == 'INCOMPLETE_STREAM'):
                        status = 'incomplete'
                    finish(status, failure)
        response.body_iterator = observed()
        return response
    return wrapped


def _warning(message):
    global _warned
    if not _warned:
        _warned = True
        print('WARNING: Conversation diagnostic log: ' + message, file=sys.stderr)


def _write():
    try:
        directory = Path(__file__).resolve().parents[1] / 'logs' / 'conversations'
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f'_{os.getpid()}.jsonl'
        with (directory / name).open('a', encoding='utf-8', buffering=1) as stream:
            while True:
                record = _queue.get()
                try:
                    if record is None:
                        return
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                finally:
                    _queue.task_done()
    except Exception as exc:
        _warning(str(exc))


def record(event, **fields):
    global _thread
    try:
        _capture(event, fields)
        with _guard:
            if _thread is None:
                _thread = threading.Thread(target=_write, name='conversation-log', daemon=True)
                _thread.start()
        if not _thread.is_alive():
            return
        entry = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                     request_id=trace_id(), pid=os.getpid(), event=event)
        entry.update(fields)
        _queue.put_nowait(entry)
    except queue.Full:
        _warning('queue full; some diagnostic records were skipped')
    except Exception as exc:
        _warning(str(exc))


def _close():
    if _thread and _thread.is_alive():
        try:
            _queue.put(None, timeout=1)
            _thread.join(timeout=3)
        except queue.Full:
            pass


atexit.register(_close)
