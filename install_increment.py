"""Install this complete-file increment with automatic backups; no data or secrets are copied."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Existing Legal_LLM root')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    target = args.project.resolve()
    if source == target:
        parser.error('Extract the increment separately; the installer needs the existing files for backup.')
    if not (target / 'backend/routers/chat.py').is_file() or not (target / 'rag/query_pipeline.py').is_file():
        parser.error('Target does not look like the current Legal_LLM project root.')
    manifest = json.loads((source / 'increment_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            parser.error('Unsafe manifest path.')
        path = source / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            parser.error('File checksum mismatch: ' + str(relative))
    backup = target / 'backups' / ('multidomain_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    previous = []
    # Complete all backups before replacing the first production file.
    for item in manifest['files']:
        relative = Path(item['path'])
        old = target / relative
        if old.exists():
            dest = backup / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, dest)
            previous.append(str(relative))
    (backup / 'installation_record.json').write_text(json.dumps({'previous_files': previous,
        'new_files': [item['path'] for item in manifest['files'] if item['path'] not in previous]}, indent=2), encoding='utf-8')
    for item in manifest['files']:
        relative = Path(item['path'])
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    print('Installed complete production files.')
    print('Backup:', backup)
    print('Next: activate your env, pip install -r requirements_admin.txt, then restart LLM/backend/admin.')
    print('Instructions:', target / 'README_MULTI_DOMAIN.md')


if __name__ == '__main__':
    main()
