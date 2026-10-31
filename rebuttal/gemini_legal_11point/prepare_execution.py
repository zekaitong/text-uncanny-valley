#!/usr/bin/env python3
"""Offline-only: seal the verified 800 inputs and four-worker execution package."""
import os
import shutil
from workflow import ROOT, PROJECT, ASSESSMENT, MODEL, RATES, CODE_FILES, historical, load, sha, save, verify, shaobj, now


def main():
    os.umask(0o077)
    if (ROOT / 'execution_manifest.json').exists():
        verify(); print('Existing sealed package verified; not regenerated. No API client.'); return
    assessment = load(ASSESSMENT / 'assessment_manifest.json')
    assert assessment['requests'] == 800 and assessment['reference_policy'].startswith('Option A;')
    legacy = historical(); legacy.assert_pins(PROJECT)
    for rel, value in assessment['artifact_sha256'].items():
        assert sha(ASSESSMENT / rel) == value
    inputs = ['requests.jsonl', 'mapping.jsonl'] + [f'references/wfr_{i:02d}.jsonl' for i in RATES]
    for rel in inputs:
        dest = ROOT / 'inputs' / rel; dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if dest.exists(): raise ValueError('Partial preparation exists; do not silently overwrite')
        shutil.copyfile(ASSESSMENT / rel, dest)
    for name in ['logs', 'attempts', 'plans', 'runs', 'locks', 'derived']:
        (ROOT / name).mkdir(exist_ok=True, mode=0o700)
    source_hashes = dict(assessment['source_hashes'])
    old_pins = load(PROJECT / 'camera_ready/gemini_batch_legal_11point/source_fingerprints.json')
    for rel, h in old_pins.items():
        path = str(PROJECT / rel) if not rel.startswith('/') else rel
        assert sha(path) == h; source_hashes[path] = h
    source_hashes[str(PROJECT / 'camera_ready/gemini_batch_legal_11point/batch_common.py')] = sha(PROJECT / 'camera_ready/gemini_batch_legal_11point/batch_common.py')
    # Frozen historical inventory is used for collection, never modified in place.
    histories = []
    for record in load(ASSESSMENT / 'historical_inventory.json'):
        i = round(record['wfr'] * 10)
        path = PROJECT / 'data/results' / (
            f'ld/gemini_3_flash_baseline/{"ori" if not i else "w" + str(i)}/results.json'
            if record['arm'] == 'unrepaired' else
            f'clean_wordseg_repair/ld/gemini-3.0-flash/wa/{"ori" if not i else i}/results.json')
        source_hashes[str(path)] = sha(path)
        histories.append({**record, 'path': str(path)})
    save(ROOT / 'inputs/historical_inventory.json', histories, exclusive=True)
    files = CODE_FILES + ['inputs/' + r for r in inputs] + ['inputs/historical_inventory.json']
    manifest = {'model': MODEL, 'requests': 800, 'rates': RATES, 'concurrency': 4,
        'sdk': 'google-genai==1.63.0', 'created_utc': now(), 'route': 'standard_generate_content',
        'reference_policy': 'Option A; repaired inputs and repaired original_context + archived omitted_index; unchanged legacy scorer',
        'generation': {'thinkingConfig': {'thinkingBudget': 0}},
        'temperature': 'omitted', 'max_output_tokens': 'omitted', 'stop_sequences': 'omitted',
        'transport_timeout_ms': 600000, 'http_attempts': 1,
        'sources_sha256': source_hashes, 'files_sha256': {rel: sha(ROOT / rel) for rel in files}}
    save(ROOT / 'execution_manifest.json', manifest, exclusive=True)
    _, mapping, requests, _ = verify()
    plan = {'plan_id': 'initial', 'created_utc': now(), 'execution_manifest_sha256': sha(ROOT / 'execution_manifest.json'),
            'parent_array_job_id': None, 'entries': [{'key': r['key'], 'attempt': 1, 'wfr': mapping[r['key']]['wfr']} for r in requests],
            'policy': 'one initial attempt, no automatic paid retry'}
    save(ROOT / 'plans/initial.json', {'plan': plan, 'sha256': shaobj(plan)}, exclusive=True)
    print('SEALED: exactly 800 Option A requests; 8 array tasks, max 4 workers; no paid activity.')


if __name__ == '__main__': main()
