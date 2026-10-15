"""Write completed database messages off the request thread; never log tokens."""
import atexit
import json
import os
from pathlib import Path
import queue
import sys
import threading
from datetime import datetime, timezone

from .timing import trace_id

_queue = queue.Queue(maxsize=1024)
_guard = threading.Lock()
_thread = None
_warned = False


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
        with _guard:
            if _thread is None:
                _thread = threading.Thread(target=_write, name='conversation-log', daemon=True)
                _thread.start()
        if not _thread.is_alive():
            return
        _queue.put_nowait(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                              request_id=trace_id(), pid=os.getpid(), event=event, **fields))
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
