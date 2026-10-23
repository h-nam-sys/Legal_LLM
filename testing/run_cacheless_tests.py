"""Run unchanged original/fresh fixtures against the live system, with answer cache bypassed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUITES = {'original_64':'qa_baseline_64_v1.json', 'fresh_48':'qa_fresh_cases_v1.json',
          'probe':'qa_generalization_probe_v1.json', 'targeted':'qa_targeted_ab_v1.json'}


def compare_run(label, run, baseline):
    performance = run.get('performance', {})
    expected = baseline.get(label, {})
    current = performance.get('uncached_excluding_first_request', {}).get('mean_seconds')
    previous = expected.get('uncached_excluding_first_mean_seconds')
    ratio = current / previous if current is not None and previous else None
    events = performance.get('answer_cache_events', 0)
    verified = (run.get('answer_cache_bypassed') is True and events > 0
                and performance.get('cache_hits') == 0
                and performance.get('cache_bypass_acknowledged') == events)
    return {'suite':label, 'turns':run['turn_count'], 'baseline_automatic_counts':expected.get('automatic_counts'),
            'baseline_equivalent_unit_checker_counts':expected.get('equivalent_unit_checker_counts'),
            'current_automatic_counts':run['counts'], 'previous_checker_counts':run.get('previous_checker_counts'), 'checker_version':run.get('checker_version',1), 'cacheless_verified':verified,
            'baseline_uncached_excluding_first_seconds':previous,
            'current_uncached_excluding_first_seconds':current,
            'latency_change_percent':(ratio - 1) * 100 if ratio is not None else None,
            'speed_review_required':ratio > 1.10 if ratio is not None else None,
            'current_performance':performance, 'runtime_options_observed':run.get('runtime_options_observed',[]),
            'note':'Automatic checks are not a manual factual grade. A faster refusal is not a quality improvement. ' +
                   ('When a verified cacheless baseline is present, compare the same complete suite. An older mixed-cache baseline only supports an uncached-subset comparison.'
                    if label=='original_64' else 'Inspect guidance_comparison.json for paired targeted results; a missing baseline is not a measured slowdown.'
                    if label=='targeted' else 'Compare the same questions and cache conditions; manual answer review is still required.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['both',*SUITES],default='both')
    parser.add_argument('--repeats',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--timeout',type=int,default=300)
    args = parser.parse_args()
    baseline_path = ROOT/'testing/fixtures/qa_cacheless_baseline_20261022.json'
    if not baseline_path.is_file():
        baseline_path = ROOT/'testing/fixtures/qa_frozen_baseline_20261022.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    comparisons, archives = [], []
    for label, filename in SUITES.items():
        if args.suite != label and not (args.suite == 'both' and label in {'original_64','fresh_48'}):
            continue
        before = set((ROOT/'qa_exports').glob('system_tests_*.zip'))
        subprocess.run([sys.executable,str(ROOT/'qa_suite.py'),'--cases',str(ROOT/'testing/fixtures'/filename),
                        '--cacheless','--require-all-domains','--repeats',str(args.repeats),
                        '--timeout',str(args.timeout)],cwd=ROOT,check=True)
        created = sorted(set((ROOT/'qa_exports').glob('system_tests_*.zip')) - before)
        if len(created) != 1:
            raise RuntimeError('Expected exactly one QA archive; avoid simultaneous test runs.')
        archives.extend(created)
        with zipfile.ZipFile(created[0]) as bundle:
            run = json.loads(bundle.read('run.json'))
            if label == 'targeted':
                from targeted_analysis import analyze
                paired = analyze(json.loads(bundle.read('qa_cases.json')), json.loads(bundle.read('results.json')))
        if label == 'targeted':
            with zipfile.ZipFile(created[0], 'a', zipfile.ZIP_DEFLATED) as target:
                target.writestr('guidance_comparison.json', json.dumps(paired, ensure_ascii=False, indent=2))
        manifest = json.loads((ROOT/'testing/fixtures'/filename).read_text(encoding='utf-8'))
        expected_turns = sum(len(c['turns']) for c in manifest['scenarios']) * args.repeats
        if run['turn_count'] != expected_turns or run.get('skipped_scenarios'):
            raise RuntimeError('Suite coverage differs from the frozen baseline. Inspect export before comparison.')
        comparison = compare_run(label,run,baseline)
        comparisons.append(comparison)
        print('\n' + json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = ROOT/'qa_exports'/('cacheless_comparison_' + stamp + '.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('comparison.json',json.dumps({'runs':comparisons,'manual_review_notes':baseline['notes']},ensure_ascii=False,indent=2))
        for archive in archives:
            bundle.write(archive,archive.name)
    print('\nSend this combined ZIP:\n' + str(destination),flush=True)
    if not all(c['cacheless_verified'] for c in comparisons):
        raise RuntimeError('Cache bypass was not verified. Check backend version and timing logs; do not accept speed figures as cacheless.')


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        raise SystemExit('Cacheless test stopped: ' + str(exc))
