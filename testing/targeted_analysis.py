"""Pair recorded answers for review; absence-language flags are NOT factual grades."""
from collections import Counter
import re
import statistics
import unicodedata


def missing_information_language(answer):
    text = unicodedata.normalize('NFD', answer.casefold().replace('đ','d'))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return bool(re.search(
        r'(?:khong|chua)\s+(?:co|tim thay|cung cap)\s+(?:thong tin|du lieu)|'
        r'(?:thong tin|du lieu)[^\n.!?]{0,100}(?:khong|chua)\s+(?:co|duoc cung cap)', text))


def analyze(manifest, results):
    cases = {case['id']:case for case in manifest['scenarios']}
    paired = {}
    for result in results:
        case = cases[result['scenario']]
        group = case.get('comparison_group')
        if not group:
            continue  # Explicit warmup stays in raw results but not the comparison.
        key = (group,result.get('repeat',1),result['turn'])
        item = paired.setdefault(key, {'group':group,'repeat':key[1],'turn':key[2], 'prompt':result['prompt']})
        variant = 'on' if case['answer_guidance'] else 'off'
        answer = result.get('actual',{}).get('stream',{}).get('answer','')
        item[variant] = {'answer':answer,'automatic_status':result['status'],
            'elapsed_seconds':result.get('elapsed_ms',0)/1000,
            'request_id':result.get('request_id'),
            'titles':result.get('observed',{}).get('titles',[]),
            'domain':result.get('observed',{}).get('domain'),
            'missing_information_language':missing_information_language(answer),
            'checks':[c for c in result.get('checks',[]) if c['status']!='PASS']}
    rows = list(paired.values())
    if any('on' not in row or 'off' not in row for row in rows):
        raise ValueError('Incomplete guidance on/off pairs')
    for row in rows:
        row['same_selected_records'] = row['on']['titles'] == row['off']['titles']
        row['same_domain'] = row['on']['domain'] == row['off']['domain']
    def summary(variant):
        values = [row[variant] for row in rows]
        durations = [v['elapsed_seconds'] for v in values]
        return {'turn_count':len(values), 'automatic_counts':dict(Counter(v['automatic_status'] for v in values)),
                'mean_seconds':statistics.mean(durations) if durations else None,
                'median_seconds':statistics.median(durations) if durations else None,
                'missing_information_language_count':sum(v['missing_information_language'] for v in values)}
    return {'note':'Identical paired questions/context chains; answer guidance alone is requested on/off. '
                   'Warmup is excluded. Alternating order reduces but does not eliminate order effects. '
                   'Missing-information language may be a correct abstention: inspect each answer, not just flag counts. '
                   'One run does not establish statistical significance or new-dataset generalization.',
            'on':summary('on'), 'off':summary('off'), 'pairs':rows}
