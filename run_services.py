"""Run the existing three services in one terminal; retain full output on disk."""
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from dotenv import load_dotenv
from rag.service_config import llm_port

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
OUTPUT_LOCK = threading.Lock()


def show(service, text):
    with OUTPUT_LOCK:
        print(f'{time.strftime("%H:%M:%S")} [{service}] {text}', flush=True)


def capture(name, process, path):
    try:
        in_traceback = False
        with path.open('a', encoding='utf-8', buffering=1) as output:
            for line in process.stdout:
                output.write(line)
                # Raw library/model progress stays on disk. Activity and problems
                # stay visible, without printing every HTTP request or timer.
                if line.lstrip().startswith('Traceback (most recent call last):'):
                    in_traceback = True
                    continue
                if in_traceback and '[activity]' in line:
                    in_traceback = False
                if in_traceback:
                    if re.match(r'^(?:[\w.]*(?:Error|Exception|ExceptionGroup)|KeyboardInterrupt|SystemExit)(?::|$)', line):
                        show(name, line.strip() + f' (full traceback: {path})')
                        in_traceback = False
                    continue
                if '[activity]' in line:
                    show(name, line.split('[activity]', 1)[1].strip())
                elif re.search(r'\b(ERROR|WARNING|CRITICAL|Traceback)\b|\w+(?:Error|Exception):', line):
                    show(name, line.strip())
    except Exception as exc:
        show(name, f'ERROR reading service output: {exc}')
        STOP.set()


def watch_ready(name, process, url):
    while not STOP.is_set() and process.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    show(name, 'Ready: ' + url)
                    return
        except Exception:
            pass
        STOP.wait(2)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # Includes uvicorn reload children; never targets unrelated Python jobs.
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        # The process-group leader may have exited before its children. taskkill
        # is best-effort and uses this launcher's original PID only.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def service_definitions():
    inference_port=llm_port()
    return [
        ('LLM', ROOT, [sys.executable, '-u', 'llm_service/main.py'], inference_port, '/v1/rag/status'),
        ('Backend', ROOT / 'backend', [sys.executable, '-u', '-m', 'uvicorn', 'main:app', '--port', '8000', '--reload'], 8000, '/health'),
        ('Admin', ROOT / 'backend', [sys.executable, '-u', '-m', 'streamlit', 'run', 'admin_dashboard.py', '--server.address', '127.0.0.1', '--server.port', '8501'], 8501, '/_stcore/health'),
    ]


def main():
    load_dotenv(ROOT / 'backend/.env')
    load_dotenv(ROOT / '.env')
    services = service_definitions()
    for name, _, _, port, _ in services:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                show(name, f'ERROR: port {port} is already occupied. Stop the old services first.')
                return 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = ROOT / 'logs' / 'services' / stamp
    directory.mkdir(parents=True)
    show('Services', 'Starting. Ctrl+C stops all three services.')
    show('Logs', str(ROOT / 'logs'))
    children = []
    readers = []
    result = 0
    try:
        for name, cwd, command, port, health in services:
            environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                       creationflags=flags, start_new_session=os.name != 'nt')
            children.append((name, process))
            reader = threading.Thread(target=capture, args=(name, process, directory / (name.lower() + '.log')), daemon=True)
            reader.start()
            readers.append(reader)
            threading.Thread(target=watch_ready, args=(name, process, f'http://127.0.0.1:{port}{health}'), daemon=True).start()
            show(name, 'Starting...')
        while not STOP.wait(0.5):
            for name, process in children:
                if process.poll() is not None:
                    show(name, f'Exited with code {process.returncode}. Full output: {directory}')
                    result = process.returncode or 1
                    STOP.set()
                    break
    except KeyboardInterrupt:
        show('Services', 'Stopping...')
    except Exception as exc:
        show('Services', f'ERROR: {exc}')
        result = 1
    finally:
        STOP.set()
        # Stop services concurrently so shutdown does not wait three timeouts.
        stoppers = [threading.Thread(target=stop_process, args=(p,)) for _, p in children]
        for thread in stoppers:
            thread.start()
        for thread in stoppers:
            thread.join()
        for thread in readers:
            thread.join(timeout=2)
        show('Services', 'Stopped.')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
