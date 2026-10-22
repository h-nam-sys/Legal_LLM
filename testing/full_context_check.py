"""Run the unchanged combined suite and label whether full-context mode was actually used."""
from pathlib import Path
import hashlib
import json
import zipfile

BASELINE={
    'backend/services/domain_router.py':'9f6633a4116ff7a5a464d7587cf70cce52b89481604ed0b75ca3ae64982b0eb3',
    'backend/services/chat_orchestrator.py':'0086956427b36e9ebd08e371343f3dab8fed325d3212080edbe380f08114e39c',
}


def annotate(archive):
    with zipfile.ZipFile(archive) as z:
        run=json.loads(z.read('run.json'))
        events=[]
        for name in z.namelist():
            if not name.startswith('logs/timing/'):
                continue
            for line in z.read(name).decode('utf-8',errors='replace').splitlines():
                try:events.append(json.loads(line.split('TIMING ',1)[1]))
                except (IndexError,ValueError):pass
    cache=[e for e in events if e.get('task')=='answer.cache']
    selections=[e for e in events if e.get('task')=='rag.evidence.diagnostic' and e.get('event')=='field_selection']
    contexts=[e for e in events if e.get('task')=='rag.evidence.diagnostic' and e.get('event')=='complete']
    hashes={name.replace('\\','/'):value for name,value in run.get('runtime_sha256',{}).items()}
    valid=bool(cache) and all(e.get('context_mode')=='full' for e in cache) and not selections
    valid=valid and bool(contexts) and all(e.get('context_mode')=='full' for e in contexts)
    valid=valid and all(hashes.get(name)==value for name,value in BASELINE.items())
    data={'experiment':'full public records without column selection',
        'configuration_verified':valid,'context_mode':'full',
        'cache_events':len(cache),'field_selection_calls':len(selections),
        'evidence_contexts':len(contexts),'baseline_runtime_sha256':BASELINE,
        'note':'Same combined test manifest. Context and prompt sizes are characters, not token counts. Field checks measure supplied evidence; answer facts still need their own checks.'}
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        z.writestr('experiment.json',json.dumps(data,ensure_ascii=False,indent=2))
    return valid

