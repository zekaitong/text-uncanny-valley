"""Offline protocol helpers. Importing this module never creates an API client."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('TEXT_UNCANNY_PROJECT_ROOT', HERE.parents[1])).resolve()
MODEL = 'gemini-3-flash-preview'
SDK_VERSION = '1.63.0'
PINS = {
    'tests/test_llms.py': '11ee63316aa0bc5a1ba9830338a3f0a7ef206f7a34f4402d6a20b73f2825ab98',
    'tests/llm_providers.py': 'c8694c6c28a712b6abf900d2fdae5141ca8012091b5ddb26240fd29cf6995309',
    'rebuttal/dictionary_repair/segment_repair_spacing_clean.py': '7b6e901f5fdb6011d202942717df0138d93e6b68a326c8fe5ffe99214fe11903',
    'run_benchmark.py': '4345aacd9b4193e65db8cd36fbc3938cd5e2492aa247c9bf04490398f8547f16',
}
DICT_SHA = '829a043cf078d1e80e886289a13823454977f442a239a859d2133ea61944aa60'
DICTIONARY = Path(os.environ.get(
    'TEXT_UNCANNY_DICTIONARY',
    HERE.parent / 'dictionary_repair' / 'en_US.dic',
)).resolve()
GENERATION = {'thinkingConfig': {'thinkingBudget': 0}}


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def object_sha(value):
    return digest(canonical(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def assert_pins(project):
    for rel, expected in PINS.items():
        if file_sha(project / rel) != expected:
            raise ValueError('Historical source changed: ' + rel)
    if file_sha(DICTIONARY) != DICT_SHA:
        raise ValueError('External dictionary changed')


def legacy_adapter(project):
    """Extract only scorer/prompt AST; never import provider or runner modules."""
    tree = ast.parse((project / 'tests/test_llms.py').read_text())
    fs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    scorer = copy.deepcopy(fs['evaluate_response'])
    prompt = copy.deepcopy(fs['process_document'])
    prefix = []
    for node in prompt.body:
        if isinstance(node, ast.Try):
            break
        prefix.append(node)
    prompt.body = prefix + [ast.Return(value=ast.Name(id='formatted_message', ctx=ast.Load()))]
    defaults = []
    for node in fs['test_model'].body:
        if isinstance(node, ast.If):
            if ast.unparse(node.test) in ('not document_type', 'not system_prompt', 'use_needle'):
                defaults.append(copy.deepcopy(node))
                if ast.unparse(node.test) == 'use_needle':
                    break
    if len(defaults) != 3:
        raise ValueError('Unrecognized historical prompt construction')
    env = dict(Counter=Counter, Dict=Dict, List=List, Any=Any)
    module = ast.fix_missing_locations(ast.Module(body=[scorer, prompt], type_ignores=[]))
    exec(compile(module, '<historical-scorer-and-prompt>', 'exec'), env)
    system_env = dict(document_type='legal_documents', system_prompt=None, use_needle=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=defaults, type_ignores=[])),
                 '<historical-system-prompt>', 'exec'), system_env)

    def user_prompt(doc):
        return env['process_document'](doc, 0, 'google', MODEL, system_env['system_prompt'],
                                       None, False, False, 'legal_documents', '\n')
    return env['evaluate_response'], system_env['system_prompt'], user_prompt


def load_repair(project):
    name = '_existing_clean_repair'
    spec = importlib.util.spec_from_file_location(
        name, project / 'rebuttal/dictionary_repair/segment_repair_spacing_clean.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    words, sources = module.load_words([str(DICTIONARY)])
    return module, module.RepairConfig(frozenset(words)), sources


def validate_wire(row):
    from google.genai import types
    if set(row) != {'key', 'request'} or not isinstance(row['key'], str):
        raise ValueError('Invalid JSONL envelope')
    r = row['request']
    if set(r) != {'contents', 'systemInstruction', 'generationConfig'}:
        raise ValueError('Unexpected wire fields')
    if r['generationConfig'] != GENERATION:
        raise ValueError('Generation settings drift')
    cfg = types.GenerateContentConfig.model_validate({
        **r['generationConfig'], 'systemInstruction': r['systemInstruction']})
    sdk = types.InlinedRequest.model_validate({'contents': r['contents'], 'config': cfg})
    dumped = sdk.config.model_dump(mode='json', exclude_none=True, by_alias=True)
    if dumped.pop('systemInstruction') != r['systemInstruction'] or dumped != GENERATION:
        raise ValueError('SDK changed configuration')
    if [c.model_dump(mode='json', exclude_none=True, by_alias=True) for c in sdk.contents] != r['contents']:
        raise ValueError('SDK changed contents')
    return sdk


def validate_artifacts(directory, check_sources=True):
    directory = Path(directory)
    m = read_json(directory / 'experiment_manifest.json')
    if m['model'] != MODEL or m['generation_config'] != GENERATION:
        raise ValueError('Manifest model/config mismatch')
    if check_sources:
        assert_pins(Path(m['project']))
    for rel, expected in m['artifact_sha256'].items():
        if file_sha(directory / rel) != expected:
            raise ValueError('Artifact modified: ' + rel)
    mappings = {r['key']: r for r in rows(directory / 'request_mapping.jsonl')}
    if len(mappings) != 2200:
        raise ValueError('Mapping must have 2200 unique rows')
    seen = set()
    by_pair = {}
    levels = Counter()
    for row in rows(directory / 'requests.jsonl'):
        validate_wire(row)
        key = row['key']
        if key in seen or key not in mappings:
            raise ValueError('Duplicate or unknown request key')
        seen.add(key)
        mp = mappings[key]
        if object_sha(row['request']) != mp['request_sha256']:
            raise ValueError('Request mapping hash mismatch')
        by_pair.setdefault(mp['pair_id'], []).append(mp)
        levels[mp['wfr']] += 1
    if seen != set(mappings) or set(levels) != {i / 10 for i in range(11)}:
        raise ValueError('Request coverage mismatch')
    if any(n != 200 for n in levels.values()) or len(by_pair) != 1100:
        raise ValueError('Pair counts mismatch')
    for pair in by_pair.values():
        if {x['condition'] for x in pair} != {'repaired', 'unrepaired'} or len(pair) != 2:
            raise ValueError('Unmatched pair')
        for field in ('document_id', 'wfr', 'source_content_sha256', 'fragmented_original_sha256',
                      'fragmented_modified_sha256', 'omission_mask_sha256', 'ground_truth_sha256',
                      'prompt_version', 'generation_config_sha256'):
            if pair[0][field] != pair[1][field]:
                raise ValueError('Pair invariant mismatch: ' + field)
    shard_keys = []
    for shard in m['shards']:
        part_keys = [r['key'] for r in rows(directory / shard['file'])]
        if len(part_keys) != shard['requests']:
            raise ValueError('Shard count mismatch')
        shard_keys.extend(part_keys)
    if len(shard_keys) != 2200 or set(shard_keys) != seen:
        raise ValueError('Shard coverage mismatch')
    return m, mappings


def manual_client():
    """Only called by user-run submission/collection/auth commands."""
    import logging
    from google import genai
    from google.genai import types
    logging.disable(logging.CRITICAL)
    # Explicit key selection avoids GOOGLE_API_KEY taking precedence. Never log it.
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise ValueError('GEMINI_API_KEY is not set in this terminal')
    return genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
        api_version='v1beta', base_url='https://generativelanguage.googleapis.com',
        retry_options=types.HttpRetryOptions(attempts=1)))


def safe_main(fn):
    try:
        fn()
    except Exception as exc:
        # Provider exceptions may contain headers/URLs. Do not serialize messages.
        print('Stopped safely: ' + type(exc).__name__ + '. No automatic retry. See README.', file=sys.stderr)
        raise SystemExit(1) from None
