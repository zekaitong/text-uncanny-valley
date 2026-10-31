#!/usr/bin/env python3
"""Offline validation, progress, collection and explicitly approved resume plans."""
import argparse
import csv
import json
import os
import statistics
import subprocess
from collections import Counter
from workflow import (ROOT, PROJECT, MODEL, RATES, verify, read_plan, historical, attempt_files,
                      load, save, sha, shaobj, now, quality, guarded_main)


def inspect_progress(mapping):
    counts = Counter(); by_rate = {i: Counter() for i in RATES}; states = {}
    for key, mp in mapping.items():
        attempts = [load(f) for f in attempt_files(key)]
        responses = [x for x in attempts if 'raw_response' in x]
        state = 'SAVED_RESPONSE' if responses else (attempts[-1]['status'] if attempts else 'NOT_ATTEMPTED')
        counts[state] += 1; by_rate[round(mp['wfr'] * 10)][state] += 1
        states[key] = (attempts, responses)
    return counts, by_rate, states


def status():
    _, mapping, _, _ = verify(); counts, rates, _ = inspect_progress(mapping)
    print(str(len(mapping)) + ' prepared requests. Saved response != valid text; use collect for quality flags.')
    print(json.dumps({'overall': dict(counts), 'by_wfr': {i / 10: dict(v) for i, v in rates.items()}}, indent=2))
    for path in sorted((ROOT / 'runs').glob('*/STOP.json')):
        print('STOP FLAG:', path.parent.name, 'Inspect attempts; do not delete flags to force retries.')


def csvout(path, records):
    if not records: return
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)


def collect():
    from google.genai import types
    _, mapping, _, refs = verify(); score, _, _ = historical().legacy_adapter(PROJECT)
    counts, _, states = inspect_progress(mapping)
    records = []; new_results = []; failures = []; usage = []; flags = Counter()
    for h in load(ROOT / 'inputs/historical_inventory.json'):
        for s in load(h['path'])['google'][MODEL]['detailed_results']:
            records.append({'wfr': h['wfr'], 'condition': h['arm'], 'document_id': s['id'],
                'legacy_f1': s['micro_f1'], 'provenance': 'historical', 'date': h['date'], 'path': h['path']})
    for key, mp in mapping.items():
        attempts, responses = states[key]
        if not responses:
            failures.append({'key': key, 'wfr': mp['wfr'], 'document_id': mp['document_id'],
                'status': attempts[-1]['status'] if attempts else 'NOT_ATTEMPTED',
                'attempt_count': len(attempts)}); continue
        # First saved response wins. Never select the best or latest answer.
        event = responses[0]; raw = event['raw_response']
        try:
            response = types.GenerateContentResponse.model_validate(raw)
            text = response.text
            result = score([text, 0, ''], refs[round(mp['wfr'] * 10)][mp['document_id']], False, delimiter='\n')
        except Exception:
            failures.append({'key': key, 'wfr': mp['wfr'], 'document_id': mp['document_id'],
                             'status': 'OFFLINE_PARSE_OR_SCORE_ERROR', 'attempt_count': len(attempts)})
            continue
        q = quality(raw); flags.update(q['flags'])
        record = {'key': key, 'wfr': mp['wfr'], 'document_id': mp['document_id'],
                  'attempt': event['attempt'], 'model_response': [text, 0, ''],
                  'legacy_evaluation': result, 'quality': q, 'has_text_value': text is not None,
                  'date': event.get('response_received_utc', event['started_utc']),
                  'raw_response_sha256': shaobj(raw),
                  'model_version': raw.get('modelVersion', raw.get('model_version')),
                  'usage_metadata': raw.get('usageMetadata', raw.get('usage_metadata'))}
        new_results.append(record)
        records.append({'wfr': mp['wfr'], 'condition': 'repaired', 'document_id': mp['document_id'],
            'legacy_f1': result['micro_f1'], 'provenance': 'new', 'date': record['date'],
            'path': str(ROOT / f'attempts/{key}/{event["attempt"]:04d}.json')})
        if record['usage_metadata'] is not None: usage.append({'key': key, 'usage_metadata': record['usage_metadata']})
    curves = []; differences = []
    for i in range(11):
        row = {'wfr': i / 10}
        groups = {arm: [r for r in records if r['wfr'] == i / 10 and r['condition'] == arm] for arm in ['unrepaired', 'repaired']}
        for arm, rs in groups.items():
            row[arm + '_n'] = len(rs); row[arm + '_expected_n'] = 100
            row[arm + '_mean_legacy_f1'] = statistics.mean(r['legacy_f1'] for r in rs) if rs else None
        left = {r['document_id']: r for r in groups['unrepaired']}
        right = {r['document_id']: r for r in groups['repaired']}
        row['paired_available_n'] = len(left.keys() & right.keys())
        qs = [r for r in new_results if r['wfr'] == i / 10]
        row['new_flagged_response_n'] = sum(bool(r['quality']['flags']) for r in qs)
        row['new_without_text_value_n'] = sum(not r['has_text_value'] for r in qs)
        for did in sorted(left.keys() & right.keys()):
            differences.append({'wfr': i / 10, 'document_id': did,
                'repaired_minus_unrepaired': right[did]['legacy_f1'] - left[did]['legacy_f1']})
        curves.append(row)
    out = ROOT / 'derived' / now().replace(':', '-'); out.mkdir(parents=True, mode=0o700)
    for name, data in [('curves.csv', curves), ('per_document_scores.csv', records),
                       ('paired_differences.csv', differences), ('missing_or_errors.csv', failures)]:
        csvout(out / name, data)
    for name, data in [('new_results.jsonl', new_results), ('usage_metadata.jsonl', usage)]:
        with (out / name).open('x') as f:
            for r in data: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    historical_saved = sum(r['provenance'] == 'historical' for r in records)
    intended_total_positions = sum(row[arm + '_expected_n'] for row in curves for arm in ['unrepaired', 'repaired'])
    summary = {'historical_saved': historical_saved, 'new_initial_expected': len(mapping), 'new_saved_api_responses': counts.get('SAVED_RESPONSE', 0),
        'new_scored_responses': len(new_results), 'new_with_usage_metadata': len(usage),
        'new_with_text_value': sum(r['has_text_value'] for r in new_results),
        'new_without_quality_flags': sum(not r['quality']['flags'] for r in new_results),
        'quality_flags': dict(flags), 'unscored_or_unattempted': len(failures),
        'states': dict(counts), 'intended_total_positions': intended_total_positions, 'actual_scored_positions': len(records),
        'scoring_policy': 'unchanged legacy scorer; returned None is treated as empty exactly as historically; quality flagged separately',
        'limitations': 'Cross-date preview alias; historical coverage may be incomplete; no zero imputation of absent API responses; no output filtering; first saved response only'}
    save(out / 'summary.json', summary, exclusive=True)
    print(json.dumps(summary, indent=2)); print('Derived snapshot:', out)


def prepare_resume(args):
    _, mapping, _, _ = verify()
    ids = {load(p)['array_job_id'] for p in (ROOT / 'runs').glob('*/binding.json')}
    result = subprocess.run(
        ['squeue', '--noheader', '--user=' + os.environ.get('USER', ''), '--format=%F'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0: raise ValueError('Cannot confirm Slurm queue state; no resume plan created')
    active = {v.strip() for v in result.stdout.splitlines()}
    if active & ids: raise ValueError('Existing execution array still active; wait for terminal status')
    _, _, states = inspect_progress(mapping)
    uncertain = set(args.retry_uncertain or [])
    if uncertain - mapping.keys(): raise ValueError('Unknown explicitly requested key')
    if (args.include_failed or uncertain) and not args.acknowledge_possible_billing:
        raise ValueError('Explicit billing-uncertainty acknowledgment required for any repeat attempt')
    entries = []; skipped = Counter()
    for key, mp in mapping.items():
        attempts, responses = states[key]
        if responses:
            skipped['has_saved_response'] += 1; continue
        if not attempts:
            attempt = 1
        elif key in uncertain or (args.include_failed and attempts[-1]['status'] == 'API_ERROR_UNCERTAIN'):
            attempt = max(r['attempt'] for r in attempts) + 1
        else:
            skipped['uncertain_not_authorized'] += 1; continue
        entries.append({'key': key, 'attempt': attempt, 'wfr': mp['wfr']})
    print('Proposed requests:', len(entries), 'never-attempted:', sum(e['attempt'] == 1 for e in entries),
          'explicit repeat attempts:', sum(e['attempt'] > 1 for e in entries), 'skipped:', dict(skipped))
    if not args.write:
        print('DRY RUN; no plan created. --write is required. No API client.'); return
    if not entries: raise ValueError('No eligible requests to resume')
    # Validate the ID using the same character policy as read_plan, without reading a nonexistent file.
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        raise ValueError('Invalid resume name')
    plan = {'plan_id': args.name, 'created_utc': now(), 'execution_manifest_sha256': sha(ROOT / 'execution_manifest.json'),
            'parent_array_job_ids': sorted(ids), 'entries': entries,
            'policy': 'Never-attempted + only explicitly approved repeats; no saved-response reruns',
            'billing_uncertainty_acknowledged': args.acknowledge_possible_billing}
    save(ROOT / f'plans/{args.name}.json', {'plan': plan, 'sha256': shaobj(plan)}, exclusive=True)
    print('Saved offline resume plan:', args.name, '; NO JOB SUBMITTED.')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('validate'); sub.add_parser('status'); sub.add_parser('collect')
    r = sub.add_parser('resume'); r.add_argument('--name', required=True); r.add_argument('--write', action='store_true')
    r.add_argument('--include-failed', action='store_true'); r.add_argument('--retry-uncertain', action='append')
    r.add_argument('--acknowledge-possible-billing', action='store_true')
    args = p.parse_args()
    if args.command == 'validate':
        verify(); plan = read_plan('initial'); assert len(plan['entries']) == 800
        print('PASS: sealed 800 requests / repaired references / code hashes; 8 tasks, max 4 workers; no API/key access.')
    elif args.command == 'status': status()
    elif args.command == 'collect': collect()
    else: prepare_resume(args)


if __name__ == '__main__': guarded_main(main)
