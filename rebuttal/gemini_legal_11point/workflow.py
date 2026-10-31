"""Frozen Option A protocol and durable local state; no API clients at import."""
from __future__ import annotations
import ast
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('TEXT_UNCANNY_PROJECT_ROOT', ROOT.parents[1])).resolve()
ASSESSMENT = Path(os.environ.get(
    'GEMINI_ASSESSMENT_ROOT',
    PROJECT / 'camera_ready/gemini_800_assessment_20261025',
)).resolve()
MODEL = 'gemini-3-flash-preview'
RATES = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION = {'thinkingConfig': {'thinkingBudget': 0}}
CODE_FILES = ['workflow.py', 'prepare_execution.py', 'worker.py', 'manage.py',
              'standard4.slurm', 'README.md', 'legacy_batch_common.py']


def now(): return datetime.now(timezone.utc).isoformat()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def shaobj(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).open() if s.strip()]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''): h.update(data)
    return h.hexdigest()


def save(path, obj, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        if exclusive:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def locked(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def historical():
    path = ROOT / 'legacy_batch_common.py'
    if sha(path) != 'aa4b9c069f7a8421d7e61429538fc45d2705b96d380518f1a2c5d92634d6701a':
        raise ValueError('Bundled historical helper changed')
    spec = importlib.util.spec_from_file_location('_standard4_legacy', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def provider_method():
    tree = ast.parse((PROJECT / 'tests/llm_providers.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GoogleProvider')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_response')
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<unchanged-google-get-response>', 'exec'), env)
    return env['get_response']


def verify():
    m = load(ROOT / 'execution_manifest.json')
    if m['model'] != MODEL or m['requests'] != 800 or m['rates'] != RATES or m['concurrency'] != 4:
        raise ValueError('Execution scope mismatch')
    if importlib.metadata.version('google-genai') != '1.63.0':
        raise ValueError('Use the existing verified abg1 SDK 1.63.0')
    for rel, expected in m['files_sha256'].items():
        if sha(ROOT / rel) != expected: raise ValueError('Execution artifact changed: ' + rel)
    for raw_path, expected in m['sources_sha256'].items():
        path = raw_path.replace('${PROJECT_ROOT}', str(PROJECT))
        if sha(path) != expected: raise ValueError('Historical source changed: ' + path)
    req = rows(ROOT / 'inputs/requests.jsonl'); mapping = {r['key']: r for r in rows(ROOT / 'inputs/mapping.jsonl')}
    if len(req) != len(mapping) or len(req) != 800: raise ValueError('Expected 800 requests')
    refs = {i: {r['_id']: r for r in rows(ROOT / f'inputs/references/wfr_{i:02d}.jsonl')} for i in RATES}
    legacy = historical(); _, system, prompt = legacy.legacy_adapter(PROJECT)
    seen = set(); levels = Counter()
    for r in req:
        key = r['key']; mp = mapping[key]; i = round(mp['wfr'] * 10)
        if key in seen or i not in RATES or mp['condition'] != 'repaired': raise ValueError('Request identity mismatch')
        seen.add(key); levels[i] += 1
        ref = refs[i][mp['document_id']]
        expected = {'contents': [{'role': 'user', 'parts': [{'text': prompt(ref)}]}],
                    'systemInstruction': {'parts': [{'text': system}]}, 'generationConfig': GENERATION}
        if r['request'] != expected or shaobj(r['request']) != mp['request_sha256'] or shaobj(ref) != mp['reference_sha256']:
            raise ValueError('Prompt/reference drift')
        legacy.validate_wire(r)
    if levels != Counter({i: 100 for i in RATES}) or seen != set(mapping): raise ValueError('WFR coverage mismatch')
    return m, mapping, req, refs


def read_plan(plan_id):
    if not plan_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in plan_id):
        raise ValueError('Invalid plan ID')
    envelope = load(ROOT / f'plans/{plan_id}.json'); plan = envelope['plan']
    if shaobj(plan) != envelope['sha256'] or plan['plan_id'] != plan_id:
        raise ValueError('Plan integrity mismatch')
    if plan['execution_manifest_sha256'] != sha(ROOT / 'execution_manifest.json'):
        raise ValueError('Plan refers to another execution manifest')
    mapping = {r['key']: r for r in rows(ROOT / 'inputs/mapping.jsonl')}
    seen = set()
    for entry in plan['entries']:
        key = entry['key']
        if key in seen or key not in mapping or entry['wfr'] != mapping[key]['wfr']:
            raise ValueError('Plan key/rate mismatch')
        if type(entry['attempt']) is not int or entry['attempt'] < 1:
            raise ValueError('Invalid attempt number')
        seen.add(key)
    return plan


def attempt_files(key, root=ROOT): return sorted((Path(root) / 'attempts' / key).glob('*.json'))
def attempt_path(key, number, root=ROOT): return Path(root) / 'attempts' / key / f'{number:04d}.json'


def quality(raw):
    candidates = raw.get('candidates') or []
    feedback = raw.get('promptFeedback', raw.get('prompt_feedback', {})) or {}
    reason = feedback.get('blockReason', feedback.get('block_reason'))
    flags = []
    if reason and reason != 'BLOCK_REASON_UNSPECIFIED': flags.append('PROMPT_BLOCKED')
    if not candidates: flags.append('NO_CANDIDATE')
    first = candidates[0] if candidates else {}
    finish = first.get('finishReason', first.get('finish_reason'))
    if finish == 'MAX_TOKENS': flags.append('TRUNCATED')
    elif finish and finish != 'STOP': flags.append('FINISH_' + str(finish))
    parts = (first.get('content') or {}).get('parts') or []
    visible = [p['text'] for p in parts if isinstance(p.get('text'), str) and not p.get('thought')]
    if not visible: flags.append('NO_TEXT_PART')
    elif not ''.join(visible).strip(): flags.append('EMPTY_TEXT')
    return {'finish_reason': finish, 'prompt_block_reason': reason, 'text_part_count': len(visible), 'flags': flags}


def run_one(request, ref, path, generate, get_response, score, metadata):
    """One attempt only; raw API response is durable before parsing or scoring."""
    import types as pytypes
    event = {'key': request['key'], 'status': 'STARTED_UNCERTAIN', 'started_utc': now(), **metadata}
    save(path, event, exclusive=True)
    start = time.monotonic()
    class Capture:
        def generate_content(self, **kwargs):
            response = generate(**kwargs)
            raw = response.model_dump(mode='json', by_alias=True, exclude_none=True, exclude={'sdk_http_response'})
            event.update(status='RESPONSE_RECEIVED', raw_response=raw, quality=quality(raw), response_received_utc=now())
            save(path, event)
            return response
    provider = pytypes.SimpleNamespace(client=pytypes.SimpleNamespace(models=Capture()))
    try:
        wire = request['request']
        response = get_response(provider, wire['systemInstruction']['parts'][0]['text'],
                                wire['contents'][0]['parts'][0]['text'], MODEL, False)
        result = score(response, ref, False, delimiter='\n')
        event.update(status='COMPLETE', model_response=response, legacy_evaluation=result)
    except Exception as exc:
        code = getattr(exc, 'code', None)
        event.update(status='LOCAL_ERROR_WITH_RAW' if 'raw_response' in event else 'API_ERROR_UNCERTAIN',
                     error_type=type(exc).__name__, error_code=code if isinstance(code, int) else None)
    event.update(ended_utc=now(), elapsed_sec=time.monotonic() - start)
    save(path, event)
    return event


def assert_compute_node():
    required = ['SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']
    if not all(os.environ.get(k, '').isdigit() for k in required):
        raise ValueError('Paid execution is allowed only inside the submitted Slurm array')
    host = socket.gethostname().split('.')[0].lower()
    if host.startswith(('fe', 'login')):
        raise ValueError('API execution on login nodes is prohibited')


def guarded_main(fn):
    try: return fn()
    except Exception as exc:
        print('STOPPED:', type(exc).__name__, '(details intentionally not printed; no automatic retry)', flush=True)
        raise SystemExit(1) from None
