"""Best-effort hardware observations. Never selects a device or alters inference."""
import ctypes
import importlib.metadata
import re
import sys
from contextlib import contextmanager
from threading import Lock

from .timing import emit


class PerformanceInformation(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
            'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
            'KernelNonpaged', 'PageSize')
    ] + [(name, ctypes.c_uint32) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]


def windows_memory():
    """System commit counters, in bytes (not psutil's process VMS)."""
    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_uint32]
    api.restype = ctypes.c_int
    if not api(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        'system_commit_used_bytes': info.CommitTotal * info.PageSize,
        'system_commit_limit_bytes': info.CommitLimit * info.PageSize,
        'system_commit_available_bytes': (info.CommitLimit - info.CommitTotal) * info.PageSize,
        'system_commit_peak_bytes': info.CommitPeak * info.PageSize,
    }


def memory_snapshot(task, event, **fields):
    data, errors = {}, []
    try:
        import psutil  # Already in the project's requirements; optional for diagnostics.
        proc = psutil.Process()
        mem = proc.memory_info()
        vm = psutil.virtual_memory()
        data.update(process_rss_bytes=mem.rss, process_vms_bytes=mem.vms,
                    ram_total_bytes=vm.total, ram_available_bytes=vm.available,
                    ram_used_percent=vm.percent)
        if hasattr(mem, 'private'):
            data['process_private_bytes'] = mem.private
    except Exception as exc:
        errors.append('process_memory:' + type(exc).__name__)
    if sys.platform == 'win32':
        try:
            data.update(windows_memory())
        except Exception as exc:
            errors.append('windows_commit:' + type(exc).__name__)
    if errors:
        data['diagnostic_errors'] = errors
    emit(task, event, **fields, **data)


@contextmanager
def memory_stage(task, **fields):
    memory_snapshot(task, 'memory_before', **fields)
    status, error, message = 'ok', None, None
    try:
        yield
    except BaseException as exc:
        status, error, message = 'error', type(exc).__name__, str(exc)[:500]
        raise
    finally:
        memory_snapshot(task, 'memory_after', status=status, error=error,
                        error_message=message, **fields)


def retrieval_device(task, model, device):
    # Deliberately does not import torch, initialize CUDA, or load another model.
    emit(task, 'configured_device', model=model, configured_device=device,
         device_is_configuration_only=True)


# A callback passed to C must stay alive until it has been unregistered.
_callback_refs = []


class NativeStartupLog:
    """Mirror startup hardware evidence to timing logs; restore native logging."""
    def __init__(self, api):
        self.api, self.callback, self.previous = api, None, None
        self.user_data = ctypes.c_void_p()
        self.buffer, self.lock = '', Lock()
        self.offloaded, self.total = None, None

    def _line(self, line):
        # Persist hardware evidence only, not arbitrary model metadata/chat templates.
        if re.search(r'vulkan|cuda|metal|rocm|sycl|offload|using device|found .*device|buffer size', line, re.I):
            emit('llm.gpu.native', 'startup_evidence', message=line[:2000])
        match = re.search(r'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers', line, re.I)
        if match:
            self.offloaded, self.total = map(int, match.groups())

    def __enter__(self):
        try:
            kind = self.api.llama_log_callback
            getter = getattr(self.api, 'llama_log_get', None)
            if getter is not None:
                self.previous = kind()
                getter(ctypes.byref(self.previous), ctypes.byref(self.user_data))
            else:
                # Older llama-cpp-python versions expose the installed Python callback here.
                from llama_cpp._logger import llama_log_callback
                self.previous = llama_log_callback
            def observe(level, text, user_data):
                try:
                    decoded = (text or b'').decode('utf-8', 'replace')
                    try:
                        sys.stderr.write(decoded)
                    except Exception:
                        pass
                    with self.lock:
                        parts = (self.buffer + decoded).split('\n')
                        self.buffer = parts.pop()[-8192:]
                        for line in parts:
                            self._line(line)
                except Exception:
                    pass  # Never propagate Python exceptions through a C callback.
            self.callback = kind(observe)
            _callback_refs.append(self.callback)
            self.api.llama_log_set(self.callback, None)
            emit('llm.gpu', 'native_capture', status='enabled')
        except Exception as exc:
            emit('llm.gpu', 'native_capture', status='unavailable', error=type(exc).__name__,
                 note='Read verbose startup output in LLM terminal; offload remains unverified.')
        return self

    def __exit__(self, *exc):
        try:
            if self.callback is not None and self.previous is not None:
                self.api.llama_log_set(self.previous, self.user_data)
                _callback_refs.remove(self.callback)
        except Exception as error:
            # Retain the callback reference if unregistering failed.
            emit('llm.gpu', 'native_restore', status='unavailable', error=type(error).__name__)
        try:
            if self.buffer:
                self._line(self.buffer)
            emit('llm.gpu', 'offload_observation',
                 observed_offloaded_layers=self.offloaded, observed_total_layers=self.total,
                 verification='native_log' if self.offloaded is not None else 'unverified')
        except Exception:
            pass
        return False


def llama_runtime(api):
    data = {'requested_n_gpu_layers': -1, 'n_ctx': 4096, 'n_threads': 8}
    try:
        data['llama_cpp_python_version'] = importlib.metadata.version('llama-cpp-python')
    except Exception:
        pass
    for name in ('llama_supports_gpu_offload', 'llama_print_system_info'):
        try:
            value = getattr(api, name)()
            data[name] = value.decode('utf-8', 'replace') if isinstance(value, bytes) else value
        except Exception as exc:
            data[name + '_diagnostic_error'] = type(exc).__name__
    emit('llm.gpu', 'runtime_capabilities', **data,
         note='Build capabilities and requested layers do not prove actual offload.')


def finish_llama_startup(model):
    model.verbose = False
    try:
        from llama_cpp._logger import set_verbose
        set_verbose(False)
    except Exception as exc:
        emit('llm.gpu', 'verbosity_restore', error=type(exc).__name__)
