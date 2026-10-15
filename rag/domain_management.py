"""Admin lifecycle actions, sharing the existing ingestion writer lock."""
from datetime import datetime, timezone
from pathlib import Path
import os
import uuid

from .domain_config import (ROOT_DIR, RAGConfigurationError, atomic_json,
                           domain_status, load_domain_config, read_json)
from .ingestion_jobs import domain_lock


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

def _state_path(root, name):
    return root / 'data/domain_lifecycle' / f'{name}.json'


def _protect_default(root, name):
    default = root / 'data/domain_profile.json'
    if default.exists() and read_json(default).get('domain_name') == name:
        raise RAGConfigurationError('Không thể tắt hoặc xóa lĩnh vực mặc định hiện tại.')


@timed('rag.domain_management.set_domain_enabled')
def set_domain_enabled(name, enabled, *, root=None):
    if type(enabled) is not bool:
        raise RAGConfigurationError('enabled must be a boolean')
    root = Path(root or ROOT_DIR).resolve()
    with domain_lock(name, root=root):
        config = load_domain_config(name, root=root, allow_inactive=True)
        if not enabled:
            _protect_default(root, name)
        atomic_json(_state_path(root, config.name), {
            'status': 'enabled' if enabled else 'disabled',
            'updated_at': datetime.now(timezone.utc).isoformat(),
        })


@timed('rag.domain_management.deletion_targets')
def deletion_targets(name, *, root=None):
    """Only domain-owned data is eligible; shared uploads/history stay untouched."""
    root = Path(root or ROOT_DIR).resolve()
    config = load_domain_config(name, root=root, allow_inactive=True)
    _protect_default(root, name)
    owned = root / 'data/domains' / name
    snapshots = root / 'data/rag_generations' / name
    # Legacy custom paths can point to shared storage. Never delete such paths.
    if config.generation == 'legacy':
        for path in (config.db_path, config.vector_dir):
            if path.exists() and not path.resolve().is_relative_to(owned.resolve()):
                raise RAGConfigurationError('Dữ liệu cũ nằm ngoài thư mục riêng của lĩnh vực; không thể xóa tự động.')
    candidates = [root / 'data/domains' / f'{name}.json',
                  root / 'data/domain_settings' / f'{name}.json', owned, snapshots]
    result = []
    for path in candidates:
        # Reject directory links, including links in ancestors.
        if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
            raise RAGConfigurationError('Không thể xóa dữ liệu qua liên kết thư mục.')
        if not path.resolve().is_relative_to(root):
            raise RAGConfigurationError('Domain data must be inside the project')
        if path.exists():
            result.append(path)
    return result


@timed('rag.domain_management.delete_domain')
def delete_domain(name, confirmation, *, root=None):
    if confirmation != name:
        raise RAGConfigurationError('Nhập chính xác mã lĩnh vực để xác nhận xóa.')
    root = Path(root or ROOT_DIR).resolve()
    with domain_lock(name, root=root):
        targets = deletion_targets(name, root=root)
        state_path = _state_path(root, name)
        previous = read_json(state_path) if state_path.exists() else None
        old_status = domain_status(name, root=root)
        archive = root / 'data/deleted_domains' / (name + '_' + uuid.uuid4().hex)
        archive.mkdir(parents=True, exist_ok=False)
        manifest = {'domain': name, 'deleted_at': datetime.now(timezone.utc).isoformat(),
                    'previous_status': old_status,
                    'paths': [p.relative_to(root).as_posix() for p in targets]}
        atomic_json(archive / 'deletion_manifest.json', manifest)
        # Block new loads before moving files. Already-running requests may finish.
        atomic_json(state_path, {'status': 'deleted', 'archive': archive.relative_to(root).as_posix(),
                                 'updated_at': manifest['deleted_at']})
        moved = []
        try:
            for source in targets:
                destination = archive / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, destination)
                moved.append((source, destination))
        except OSError as exc:
            # Windows may refuse to move an open DB/index; undo completed moves.
            try:
                for source, destination in reversed(moved):
                    os.rename(destination, source)
            except OSError as rollback_error:
                raise RAGConfigurationError('Xóa bị gián đoạn; lĩnh vực đã bị chặn. Giữ bản lưu tại ' + str(archive)) from rollback_error
            if previous is None:
                state_path.unlink(missing_ok=True)
            else:
                atomic_json(state_path, previous)
            raise RAGConfigurationError('Không thể di chuyển dữ liệu. Dữ liệu đã được hoàn lại; dừng backend rồi thử lại. Bản lưu: ' + str(archive)) from exc
        return archive
