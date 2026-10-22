"""One entry point for all, original, novel, smoke and review chat tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import qa_suite as qa

ASSETS=Path(__file__).resolve().parent/'fixtures'
FIXTURE=ASSETS/'xuong_nho'


def preflight():
    """Verify existing ingestion against the oracle, without importing data again."""
    from rag.domain_config import load_domain_config,identifier
    profile=json.loads((FIXTURE/'xuong_nho_profile.json').read_text(encoding='utf-8'))
    source=json.loads((FIXTURE/'xuong_nho_dataset.json').read_text(encoding='utf-8'))
    config=load_domain_config('maker_lab_test',root=ROOT)
    for key in ('primary_entity','schema_mapping','embedding_fields','excluded_fields','source_id_field'):
        if config.profile.get(key)!=profile[key]:
            raise RuntimeError('Active maker_lab_test profile differs for '+key+'. No data changed.')
    keys=[identifier(k) for k in config.schema]
    columns=', '.join('"'+key+'"' for key in keys)
    table=identifier(config.table)
    with sqlite3.connect(config.db_path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        actual=[dict(row) for row in db.execute(f'SELECT {columns} FROM "{table}"')]
    expected=[{key:row[label] for key,label in config.schema.items()} for row in source]
    to_key=lambda row:tuple(str(row.get(key) or '') for key in keys)
    if sorted(map(to_key,actual))!=sorted(map(to_key,expected)):
        raise RuntimeError('Active maker_lab_test rows differ from the test reference. No data changed. Send this message before reimporting anything.')
    return {'domain':config.name,'generation':config.generation,'revision':config.revision,
        'fixture_rows':len(actual),'dataset_sha256':hashlib.sha256((FIXTURE/'xuong_nho_dataset.json').read_bytes()).hexdigest(),
        'note':'Read-only comparison with the already-ingested snapshot; no database edits or reingestion.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['all','original','novel','smoke','review'],default='all')
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--timeout',type=int,default=300)
    parser.add_argument('--repeats',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--verify-full-context',action='store_true')
    args=parser.parse_args()
    metadata=preflight() if args.suite in {'all','novel'} else None
    cases=ASSETS/('qa_comprehensive_cases.json' if args.suite=='all' else 'qa_cases.json')
    if args.suite=='novel':cases=FIXTURE/'qa_novel_cases.json'
    before=set((ROOT/'qa_exports').glob('system_tests_*.zip'))
    previous=sys.argv[:]
    try:
        sys.argv=['qa_suite.py','--cases',str(cases),'--require-all-domains',
            '--base-url',args.base_url,'--timeout',str(args.timeout),'--repeats',str(args.repeats)]
        if args.suite in {'smoke','review'}:sys.argv+=['--suite',args.suite]
        qa.main()
    finally:
        sys.argv=previous
    created=set((ROOT/'qa_exports').glob('system_tests_*.zip'))-before
    if len(created)!=1:raise RuntimeError('Cannot identify this test run archive')
    archive=created.pop()
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        z.writestr('testing_run.json',json.dumps({'suite':args.suite,
            'cases_sha256':hashlib.sha256(cases.read_bytes()).hexdigest(),
            'runtime_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('run_services.py','llm_service/main.py','rag/service_config.py')},
            'note':'The default all suite is the existing 64-turn chat regression, not a UI/concurrency/MCP acceptance test.'},indent=2))
        if metadata:
            z.writestr('novel_preflight.json',json.dumps(metadata,ensure_ascii=False,indent=2))
            for name in ('xuong_nho_dataset.json','xuong_nho_profile.json'):
                z.write(FIXTURE/name,'novel_dataset/'+name)
    if args.verify_full_context:
        from testing.full_context_check import annotate
        if not annotate(archive):
            raise RuntimeError('Results saved, but full-context verification failed. Send the ZIP for inspection.')
    print('Send this ZIP:\n'+str(archive.resolve()))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,sqlite3.Error) as exc:
        raise SystemExit('Comprehensive test stopped: '+str(exc))
