#!/usr/bin/env python3
"""User-run ordinary API worker. Payment requires --run and a Slurm compute node."""
import argparse
import fcntl
import logging
import os
import time
from workflow import (ROOT, RATES, MODEL, now, load, save, locked, verify, read_plan,
    historical, provider_method, assert_compute_node, attempt_path, run_one, guarded_main)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--index', type=int, required=True)
    p.add_argument('--plan', default='initial'); a = p.parse_args()
    if not a.run:
        print('DRY RUN: no key access, client, or API request. Use manage.py validate for offline checks.'); return
    assert_compute_node()
    if not 0 <= a.index < 8: raise ValueError('Array task index must be 0..7')
    os.umask(0o077)
    manifest, mapping, requests, refs = verify(); plan = read_plan(a.plan)
    array_id = os.environ['SLURM_ARRAY_JOB_ID']; rate = RATES[a.index]
    rundir = ROOT / 'runs' / a.plan; rundir.mkdir(exist_ok=True, mode=0o700)
    stop = rundir / 'STOP.json'
    with locked(rundir / 'bind.lock'):
        binding = rundir / 'binding.json'
        if binding.exists():
            if load(binding)['array_job_id'] != array_id: raise ValueError('This plan is already bound to another array; use an approved resume plan')
        else: save(binding, {'array_job_id': array_id, 'plan': a.plan, 'bound_utc': now()}, exclusive=True)
    selected = {e['key']: e for e in plan['entries'] if round(e['wfr'] * 10) == rate}
    task_status = rundir / f'task_{a.index}.json'
    with locked(ROOT / f'locks/wfr_{rate:02d}.lock', blocking=False):
        if stop.exists(): raise ValueError('Run stopped after an earlier error; inspect status and prepare a resume plan')
        if not selected:
            save(task_status, {'state': 'NO_REQUESTS', 'plan': a.plan, 'wfr': rate / 10}); print('No requests for this task.'); return
        # Four cross-process slots are an additional guard beyond Slurm's %4 throttle.
        slotfile = None; deadline = time.monotonic() + 3600
        try:
            while slotfile is None:
                if stop.exists() or time.monotonic() > deadline: raise ValueError('No available worker slot or run stopped')
                for n in range(4):
                    f = (ROOT / f'locks/api_slot_{n}.lock').open('a')
                    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: f.close(); continue
                    slotfile = f; break
                if slotfile is None: time.sleep(1)
            logging.disable(logging.CRITICAL)
            # First and only credential access/client construction: gated paid execution.
            key = os.environ.get('GEMINI_API_KEY')
            if not key: raise ValueError('GEMINI_API_KEY missing from submitted environment')
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
                api_version='v1beta', base_url='https://generativelanguage.googleapis.com',
                timeout=manifest['transport_timeout_ms'], retry_options=types.HttpRetryOptions(attempts=1)))
            del key
            score, _, _ = historical().legacy_adapter(__import__('workflow').PROJECT)
            get_response = provider_method()
            save(task_status, {'state': 'RUNNING', 'plan': a.plan, 'wfr': rate / 10,
                               'slurm_array_job_id': array_id, 'started_utc': now()})
            for request in requests:
                entry = selected.get(request['key'])
                if entry is None: continue
                if stop.exists(): raise ValueError('Shared stop flag set; no further requests sent')
                mp = mapping[request['key']]; path = attempt_path(request['key'], entry['attempt'])
                if path.exists():
                    print('SKIP_RECORDED', request['key'], 'attempt', entry['attempt'], flush=True); continue
                # Never retry a key with a previously saved API response, even under a later plan.
                from workflow import attempt_files
                if any('raw_response' in load(f) for f in attempt_files(request['key'])):
                    print('SKIP_HAS_RESPONSE', request['key'], flush=True); continue
                try:
                    event = run_one(request, refs[rate][mp['document_id']], path,
                        client.models.generate_content, get_response, score,
                        {'plan': a.plan, 'attempt': entry['attempt'], 'wfr': mp['wfr'], 'document_id': mp['document_id'],
                         'array_job_id': array_id, 'request_sha256': mp['request_sha256']})
                except FileExistsError:
                    print('SKIP_CLAIMED', request['key'], flush=True); continue
                print(request['key'], event['status'], round(event['elapsed_sec'], 2), 'sec', flush=True)
                if event['status'] == 'API_ERROR_UNCERTAIN':
                    save(stop, {'key': request['key'], 'error_type': event['error_type'],
                                'error_code': event.get('error_code'), 'stopped_utc': now()})
                    raise ValueError('API error; stopped shared run without retry')
            save(task_status, {'state': 'FINISHED_TASK', 'plan': a.plan, 'wfr': rate / 10, 'ended_utc': now()})
        finally:
            if slotfile is not None: slotfile.close()


if __name__ == '__main__': guarded_main(main)
