#!/usr/bin/env python3
"""Comprehensive mechanistic probe v2 for Text Uncanny Valley.

This version uses the project's real, pre-generated word-fragmentation JSONL
files for Legal Documents, GitHub PRs, and GSM8K-style Math Reasoning. It does
not re-sample fragmentation randomness: spans are aligned from the clean line
to the fragmented line by skipping inserted spaces.

The primary word-recovery signal is the prefix-position logit lens. The
last-fragment position is retained only as a diagnostic signal. Hidden-state
divergence is computed against the clean baseline at the corresponding prefix
position, caching only one vector per layer to keep memory bounded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    path: str
    revision: str
    n_docs_per_domain_per_wfr: int = 30
    capture_attentions: bool = True
    dtype_name: str = "bfloat16"
    single_gpu_device_map: bool = False
    low_cpu_mem_usage: bool = False
    needs_awq_overlay: bool = False
    attn_implementation: str | None = None


MODEL_SPECS = {
    "qwen2.5-7b": ModelSpec(
        display_name="Qwen2.5-7B",
        path="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796",
    ),
    "qwen2.5-7b-instruct": ModelSpec(
        display_name="Qwen2.5-7B-Instruct",
        path="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "llama-3.1-8b": ModelSpec(
        display_name="Llama-3.1-8B",
        path="meta-llama/Llama-3.1-8B",
        revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    ),
    "llama-3.1-8b-instruct": ModelSpec(
        display_name="Llama-3.1-8B-Instruct",
        path="meta-llama/Llama-3.1-8B-Instruct",
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "gemma-3-4b-it": ModelSpec(
        display_name="Gemma-3-4B-IT",
        path="google/gemma-3-4b-it",
        revision="093f9f388b31de276ce2de164bdc2081324b9767",
    ),
    "gpt-oss-20b": ModelSpec(
        display_name="GPT-OSS-20B",
        path="openai/gpt-oss-20b",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        n_docs_per_domain_per_wfr=20,
    ),
    "qwen3-30b-a3b": ModelSpec(
        display_name="Qwen3-30B-A3B",
        path="Qwen/Qwen3-30B-A3B",
        revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        n_docs_per_domain_per_wfr=20,
    ),
    "llama-3.1-70b-awq-int4": ModelSpec(
        display_name="Llama-3.1-70B-Instruct-AWQ-INT4",
        path="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        revision="2123003760781134cfc31124aa6560a45b491fdf",
        n_docs_per_domain_per_wfr=15,
        capture_attentions=False,
        dtype_name="float16",
        single_gpu_device_map=True,
        low_cpu_mem_usage=True,
        needs_awq_overlay=True,
        attn_implementation="sdpa",
    ),
}


@dataclass(frozen=True)
class DomainSpec:
    display_name: str
    clean_file: str
    rate_files: dict[float, str]
    case_sensitive: bool = False
    used_data_path: str | None = None


def rate_file(prefix: str, suffix: str, i: int) -> str:
    return f"{prefix}{i}{suffix}"


DOMAINS = {
    "ld": DomainSpec(
        display_name="Legal Documents",
        clean_file="data/legal_docs.jsonl",
        rate_files={0.0: "data/legal_docs.jsonl", **{i / 10.0: rate_file("data/legal_docs_w", ".jsonl", i) for i in range(1, 11)}},
        used_data_path="data/results/ld/claude-haiku-4-5/wa/ori/used_data.jsonl",
    ),
    "gp": DomainSpec(
        display_name="GitHub PRs",
        clean_file="data/github_prs_100.jsonl",
        rate_files={0.0: "data/github_prs_100.jsonl", **{i / 10.0: rate_file("data/github_prs_100_wa", ".jsonl", i) for i in range(1, 11)}},
        case_sensitive=True,
    ),
    "gsm": DomainSpec(
        display_name="Math Reasoning",
        clean_file="data/gsm8k_n80.jsonl",
        rate_files={0.0: "data/gsm8k_n80.jsonl", **{i / 10.0: rate_file("data/gsm8k_n80_wa", ".jsonl", i) for i in range(1, 11)}},
    ),
}


TARGET_WORDS_BY_DOMAIN = {
    "ld": [
        "agreement", "parties", "including", "written", "accordance",
        "effective", "provided", "rights", "obligations", "notice",
        "provisions", "otherwise", "pursuant", "business", "information",
        "reasonable", "termination", "property", "services", "authority",
    ],
    "gp": [
        "dependencies", "peerDependencies", "strictEqual", "assertEquals",
        "createTypeChecker", "TypeCheckerHost", "writeFileSync", "startsWith",
        "toString", "to_string", "completefuzzycollect", "electron_api_web_contents",
        "WebContents", "MaybeLocal", "string_view", "static_cast", "push_back",
        "getCurrentWindow", "loadConfiguration", "validateInputParameters",
        "refreshAccessToken", "handleSubmitFormData", "computeChecksumValue",
        "isAuthenticated",
    ],
    "gsm": [
        "minutes", "bought", "therefore", "school", "amount", "second",
        "number", "remaining", "another", "together", "dollars", "people",
        "spends", "students", "calculate", "multiply", "pounds", "divide",
        "combined", "average", "altogether", "additional", "subtract",
    ],
}


STOPWORDS = set(
    """
    about above after again against almost along already also although always
    among another around because before being below between both during each
    enough every further here however itself just more most much other otherwise
    over same several should since some still such than that their them then
    there these they this those through under until very what when where which
    while without would could
    """.split()
)

GITHUB_LOW_VALUE_IDENTIFIERS = {
    "Illuminate", "Dispatcher", "Description", "Environment", "Diagnostics",
    "TypeScript", "Copyright", "Function", "Constructor", "Parameters",
    "Arguments", "Returns", "Exception", "Examples", "Package",
}

WFR_LEVELS = [round(i / 10.0, 1) for i in range(11)]
DEFAULT_OUTPUT_DIR = "outputs/mechanistic_probe_v2"
DEFAULT_MAX_CONTEXT_CHARS = 700


@dataclass(frozen=True)
class ProbeExample:
    example_id: int
    domain_key: str
    doc_id: int
    line_idx: int
    word: str
    surface_word: str
    orig_start: int
    orig_end: int
    orig_line: str
    source: str


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_used_doc_ids(spec: DomainSpec, n_rows: int) -> list[int]:
    if spec.used_data_path:
        path = REPO_ROOT / spec.used_data_path
        if path.exists():
            ids = []
            with path.open("r", encoding="utf-8") as f:
                for fallback_idx, line in enumerate(f):
                    row = json.loads(line)
                    value = row.get("_id", row.get("id", fallback_idx))
                    try:
                        value = int(value)
                    except Exception:
                        continue
                    if 0 <= value < n_rows:
                        ids.append(value)
            if ids:
                return ids
    return list(range(n_rows))


def find_seed_match(line: str, target_words: list[str], case_sensitive: bool):
    for word in target_words:
        flags = 0 if case_sensitive else re.IGNORECASE
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", flags)
        match = pat.search(line)
        if match:
            return word, match
    return None, None


def auto_candidate_match(line: str, domain_key: str):
    if domain_key == "gp":
        candidates = re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{7,}\b", line)
        best = None
        best_score = -1
        for match in candidates:
            token = match.group(0)
            if token.isupper() or token in GITHUB_LOW_VALUE_IDENTIFIERS:
                continue
            if not re.search(r"[a-z][A-Z]|_", token):
                continue
            score = len(token)
            if "_" in token:
                score += 8
            if re.search(r"[a-z][A-Z]", token):
                score += 8
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                score += 5
            if best is None or score > best_score:
                best = match
                best_score = score
        if best:
            return best.group(0), best
        return None, None

    candidates = re.finditer(r"\b[A-Za-z][A-Za-z]{5,}\b", line)
    best = None
    for match in candidates:
        token = match.group(0).lower()
        if token in STOPWORDS:
            continue
        if best is None or len(token) > len(best.group(0)):
            best = match
    if best:
        return best.group(0).lower(), best
    return None, None


def build_probe_examples(domain_key: str, n_docs: int, allow_auto_targets: bool = True) -> list[ProbeExample]:
    spec = DOMAINS[domain_key]
    rows = read_jsonl(REPO_ROOT / spec.clean_file)
    doc_ids = load_used_doc_ids(spec, len(rows))
    targets = TARGET_WORDS_BY_DOMAIN[domain_key]
    examples: list[ProbeExample] = []
    used_docs: set[int] = set()

    def ordered_lines(lines: list[str]):
        if domain_key != "gp":
            return list(enumerate(lines))
        code_lines = [
            (idx, line)
            for idx, line in enumerate(lines)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        non_header_lines = [
            (idx, line)
            for idx, line in enumerate(lines)
            if not line.startswith(("diff --git", "index ", "@@", "+++", "---"))
        ]
        seen = set()
        ordered = []
        for item in code_lines + non_header_lines + list(enumerate(lines)):
            if item[0] in seen:
                continue
            ordered.append(item)
            seen.add(item[0])
        return ordered

    for doc_id in doc_ids:
        if len(examples) >= n_docs:
            break
        if doc_id in used_docs or doc_id >= len(rows):
            continue
        lines = rows[doc_id].get("lines", [])
        found = None
        for line_idx, line in ordered_lines(lines):
            word, match = find_seed_match(line, targets, spec.case_sensitive)
            if match is not None:
                found = (line_idx, line, word, match, "target_list")
                break
        if found is None and allow_auto_targets:
            for line_idx, line in ordered_lines(lines):
                word, match = auto_candidate_match(line, domain_key)
                if match is not None:
                    found = (line_idx, line, word, match, "auto")
                    break
        if found is None:
            continue
        line_idx, line, word, match, source = found
        examples.append(
            ProbeExample(
                example_id=len(examples),
                domain_key=domain_key,
                doc_id=doc_id,
                line_idx=line_idx,
                word=word,
                surface_word=match.group(0),
                orig_start=match.start(),
                orig_end=match.end(),
                orig_line=line,
                source=source,
            )
        )
        used_docs.add(doc_id)

    if len(examples) < n_docs:
        raise RuntimeError(f"{spec.display_name}: found {len(examples)} examples, need {n_docs}.")
    return examples


def map_orig_span_to_fragmented(orig: str, frag: str, start: int, end: int) -> tuple[int, int] | None:
    mapping: dict[int, int] = {}
    f = 0
    for o, ch in enumerate(orig):
        while f < len(frag) and frag[f] != ch:
            if frag[f] == " " and ch != " ":
                f += 1
                continue
            return None
        if f >= len(frag):
            return None
        mapping[o] = f
        f += 1
    if start not in mapping or end - 1 not in mapping:
        return None
    return mapping[start], mapping[end - 1] + 1


def context_around(text: str, start: int, end: int, max_chars: int) -> tuple[str, int, int] | None:
    target_len = end - start
    if target_len > max_chars:
        return None
    left_budget = max(0, (max_chars - target_len) // 2)
    ctx_start = max(0, start - left_budget)
    ctx_end = min(len(text), ctx_start + max_chars)
    if end > ctx_end:
        ctx_start = max(0, end - max_chars)
        ctx_end = min(len(text), ctx_start + max_chars)
    if start < ctx_start or end > ctx_end:
        return None
    return text[ctx_start:ctx_end], start - ctx_start, end - ctx_start


def find_target_token_span_by_offset(offsets: Iterable[Iterable[int]], char_start: int, char_end: int):
    token_start, token_end = None, None
    for i, pair in enumerate(offsets):
        s, e = int(pair[0]), int(pair[1])
        if s == e:
            continue
        if s < char_end and e > char_start:
            if token_start is None:
                token_start = i
            token_end = i + 1
    if token_start is None:
        return None
    return token_start, token_end


def tokenize_locate(tokenizer, text: str, char_start: int, char_end: int):
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    span = find_target_token_span_by_offset(offsets, char_start, char_end)
    if span is None:
        return None
    prefix_pos = span[0] - 1
    if prefix_pos < 0:
        return None
    return enc, offsets, span, prefix_pos


def prepare_clean_item(tokenizer, ex: ProbeExample, max_context_chars: int):
    ctx = context_around(ex.orig_line, ex.orig_start, ex.orig_end, max_context_chars)
    if ctx is None:
        return None, "clean_context_truncates_target"
    text, char_start, char_end = ctx
    located = tokenize_locate(tokenizer, text, char_start, char_end)
    if located is None:
        return None, "clean_span_or_prefix_none"
    enc, offsets, span, prefix_pos = located
    target_first_token_id = int(enc["input_ids"][0, span[0]].item())
    return {
        "text": text,
        "char_start": char_start,
        "char_end": char_end,
        "enc": enc,
        "target_span": span,
        "prefix_pos": prefix_pos,
        "target_first_token_id": target_first_token_id,
        "target_first_token_text": tokenizer.decode([target_first_token_id]),
    }, None


def prepare_fragmented_item(tokenizer, rows_by_rate: dict[float, list[dict]], ex: ProbeExample, rate: float, max_context_chars: int):
    rows = rows_by_rate[rate]
    if ex.doc_id >= len(rows):
        return None, "doc_id_out_of_range"
    try:
        frag_line = rows[ex.doc_id]["lines"][ex.line_idx]
    except Exception:
        return None, "line_missing"
    mapped = map_orig_span_to_fragmented(ex.orig_line, frag_line, ex.orig_start, ex.orig_end)
    if mapped is None:
        return None, "alignment_failed"
    frag_start, frag_end = mapped
    ctx = context_around(frag_line, frag_start, frag_end, max_context_chars)
    if ctx is None:
        return None, "fragment_context_truncates_target"
    text, char_start, char_end = ctx
    located = tokenize_locate(tokenizer, text, char_start, char_end)
    if located is None:
        return None, "fragment_span_or_prefix_none"
    enc, offsets, span, prefix_pos = located
    return {
        "text": text,
        "char_start": char_start,
        "char_end": char_end,
        "enc": enc,
        "target_span": span,
        "prefix_pos": prefix_pos,
        "fragmented_surface": text[char_start:char_end],
    }, None


def get_final_norm(model):
    candidates = [
        "model.norm",
        "model.model.norm",
        "model.language_model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
        "norm",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj, path
        except AttributeError:
            continue
    return torch.nn.Identity(), "IDENTITY_FALLBACK_CHECK_THIS"


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def move_encoding(enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def logit_lens_at_position(model, final_norm, hidden_states, position: int, target_token_id: int):
    lm_head = model.get_output_embeddings()
    ranks: list[int] = []
    probs: list[float] = []
    for layer_hidden in hidden_states:
        h = layer_hidden[0, position, :].unsqueeze(0)
        h = final_norm(h)
        logits = lm_head(h).squeeze(0).float()
        target_logit = logits[target_token_id]
        rank = int((logits > target_logit).sum().item())
        prob = float(torch.exp(target_logit - torch.logsumexp(logits, dim=-1)).item())
        ranks.append(rank)
        probs.append(prob)
    return ranks, probs


@torch.no_grad()
def attention_entropy_at_span(attentions, target_span, n_hidden_layers: int) -> list[float]:
    entropies = [math.nan]
    start, end = target_span
    for layer_attn in attentions:
        probs = layer_attn[0, :, start:end, :].float().clamp_min(1e-12)
        ent = -(probs * probs.log()).sum(dim=-1)
        entropies.append(float(ent.mean().item()))
    if len(entropies) < n_hidden_layers:
        entropies.extend([math.nan] * (n_hidden_layers - len(entropies)))
    return entropies[:n_hidden_layers]


@torch.no_grad()
def clean_prefix_vectors(model, enc, prefix_pos: int):
    device = model_device(model)
    with torch.no_grad():
        out = model(**move_encoding(enc, device), output_hidden_states=True, output_attentions=False, use_cache=False)
    vectors = torch.stack([h[0, prefix_pos, :].detach().float().cpu() for h in out.hidden_states])
    del out
    return vectors


@torch.no_grad()
def hidden_state_divergence(hidden_states_frag, clean_vectors: torch.Tensor, prefix_pos: int) -> list[float]:
    n = min(len(hidden_states_frag), clean_vectors.shape[0])
    divs: list[float] = []
    for i in range(n):
        a = hidden_states_frag[i][0, prefix_pos, :].detach().float().cpu()
        b = clean_vectors[i]
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        divs.append(1.0 - cos)
    return divs


def append_jsonl_many(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def load_completed_result_keys(path: Path) -> set[tuple[str, float, int]]:
    completed: set[tuple[str, float, int]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            try:
                completed.add((str(row["domain_key"]), round(float(row["word_fragmentation_rate"]), 1), int(row["example_id"])))
            except Exception:
                continue
    return completed


def load_model(model_key: str, spec: ModelSpec):
    override_key = "MODEL_PATH_" + re.sub(r"[^A-Za-z0-9]", "_", model_key).upper()
    model_path = os.environ.get(override_key, spec.path)
    revision = None if override_key in os.environ else spec.revision
    local_only = os.environ.get("HF_ALLOW_DOWNLOAD", "0") != "1"
    print(f"[load] {model_key} path={model_path} revision={revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=local_only,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "revision": revision,
        "local_files_only": local_only,
        "trust_remote_code": True,
        "dtype": dtype_from_name(spec.dtype_name),
        "use_safetensors": True,
    }
    if spec.attn_implementation:
        model_kwargs["attn_implementation"] = spec.attn_implementation
    elif spec.capture_attentions:
        model_kwargs["attn_implementation"] = "eager"
    if spec.single_gpu_device_map:
        model_kwargs["device_map"] = {"": 0}
    if spec.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception as exc:
        if "attn_implementation" not in model_kwargs:
            raise
        print(f"[load-retry] {model_key} without attn_implementation=eager after {type(exc).__name__}: {exc}", flush=True)
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not spec.single_gpu_device_map:
        model.to("cuda")
    model.eval()
    return model, tokenizer


def run_for_model(model_key: str, spec: ModelSpec, domain_keys: list[str], rates: list[float], output_dir: Path, args):
    model, tokenizer = load_model(model_key, spec)
    final_norm, norm_path = get_final_norm(model)
    print(f"[norm] {model_key} final_norm resolved via: {norm_path}", flush=True)

    rows_jsonl = output_dir / f"probe_results_v2_{model_key}.jsonl"
    skips_csv = output_dir / f"probe_skips_v2_{model_key}.csv"
    completed = load_completed_result_keys(rows_jsonl)
    clean_cache: dict[tuple[str, int], tuple[torch.Tensor, dict]] = {}

    for domain_key in domain_keys:
        domain = DOMAINS[domain_key]
        n_docs = args.n_docs_override if args.n_docs_override is not None else spec.n_docs_per_domain_per_wfr
        if args.smoke:
            n_docs = 1
        examples = build_probe_examples(domain_key, n_docs, allow_auto_targets=True)
        examples_path = output_dir / f"probe_examples_v2_{model_key}_{domain_key}.json"
        with examples_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in examples], f, indent=2, ensure_ascii=False)

        rows_by_rate = {rate: read_jsonl(REPO_ROOT / domain.rate_files[rate]) for rate in rates}
        for rate in rates:
            done_examples = 0
            for ex in examples:
                key = (domain_key, round(rate, 1), ex.example_id)
                if key in completed:
                    done_examples += 1
                    continue

                skip_base = {
                    "model_key": model_key,
                    "model": spec.display_name,
                    "domain_key": domain_key,
                    "domain": domain.display_name,
                    "word_fragmentation_rate": rate,
                    "example_id": ex.example_id,
                    "doc_id": ex.doc_id,
                    "line_idx": ex.line_idx,
                    "word": ex.word,
                    "surface_word": ex.surface_word,
                }
                clean_key = (domain_key, ex.example_id)
                if clean_key not in clean_cache:
                    clean_item, clean_err = prepare_clean_item(tokenizer, ex, args.max_context_chars)
                    if clean_err:
                        append_csv_rows(skips_csv, [{**skip_base, "reason": clean_err}], list(skip_base.keys()) + ["reason"])
                        continue
                    try:
                        vectors = clean_prefix_vectors(model, clean_item["enc"], clean_item["prefix_pos"])
                    except Exception as exc:
                        append_csv_rows(skips_csv, [{**skip_base, "reason": f"clean_forward_{type(exc).__name__}: {exc}"}], list(skip_base.keys()) + ["reason"])
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    clean_cache[clean_key] = (vectors, clean_item)

                clean_vectors, clean_item = clean_cache[clean_key]
                frag_item, frag_err = prepare_fragmented_item(tokenizer, rows_by_rate, ex, rate, args.max_context_chars)
                if frag_err:
                    append_csv_rows(skips_csv, [{**skip_base, "reason": frag_err}], list(skip_base.keys()) + ["reason"])
                    continue

                try:
                    device = model_device(model)
                    with torch.no_grad():
                        out = model(
                            **move_encoding(frag_item["enc"], device),
                            output_hidden_states=True,
                            output_attentions=spec.capture_attentions,
                            use_cache=False,
                        )
                    target_id = int(clean_item["target_first_token_id"])
                    prefix_pos = int(frag_item["prefix_pos"])
                    span_start, span_end = frag_item["target_span"]
                    last_pos = span_end - 1
                    ranks_prefix, probs_prefix = logit_lens_at_position(model, final_norm, out.hidden_states, prefix_pos, target_id)
                    ranks_last, probs_last = logit_lens_at_position(model, final_norm, out.hidden_states, last_pos, target_id)
                    divs = hidden_state_divergence(out.hidden_states, clean_vectors, prefix_pos)
                    if spec.capture_attentions and getattr(out, "attentions", None) is not None:
                        entropies = attention_entropy_at_span(out.attentions, frag_item["target_span"], len(out.hidden_states))
                    else:
                        entropies = [math.nan] * len(out.hidden_states)
                except Exception as exc:
                    append_csv_rows(skips_csv, [{**skip_base, "reason": f"fragment_forward_{type(exc).__name__}: {exc}"}], list(skip_base.keys()) + ["reason"])
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                out_rows: list[dict] = []
                n_layers = min(len(ranks_prefix), len(ranks_last), len(divs), len(entropies))
                for layer_idx in range(n_layers):
                    out_rows.append(
                        {
                            "model_key": model_key,
                            "model": spec.display_name,
                            "domain_key": domain_key,
                            "domain": domain.display_name,
                            "word_fragmentation_rate": rate,
                            "example_id": ex.example_id,
                            "doc_id": ex.doc_id,
                            "line_idx": ex.line_idx,
                            "word": ex.word,
                            "surface_word": ex.surface_word,
                            "target_source": ex.source,
                            "fragmented_surface": frag_item["fragmented_surface"],
                            "target_first_token_id": target_id,
                            "target_first_token_text": clean_item["target_first_token_text"],
                            "target_span_start": span_start,
                            "target_span_end": span_end,
                            "target_span_len": span_end - span_start,
                            "prefix_position": prefix_pos,
                            "last_fragment_position": last_pos,
                            "clean_prefix_position": clean_item["prefix_pos"],
                            "layer": layer_idx,
                            "recovery_rank_prefix": ranks_prefix[layer_idx],
                            "recovery_prob_prefix": probs_prefix[layer_idx],
                            "recovery_rank_last": ranks_last[layer_idx],
                            "recovery_prob_last": probs_last[layer_idx],
                            "attn_entropy": entropies[layer_idx],
                            "hidden_state_divergence": divs[layer_idx],
                            "final_norm_path": norm_path,
                            "capture_attentions": spec.capture_attentions,
                            "quantized": spec.needs_awq_overlay,
                        }
                    )
                append_jsonl_many(rows_jsonl, out_rows)
                completed.add(key)
                done_examples += 1
                print(
                    f"[done] {model_key} domain={domain_key} rate={rate:.1f} "
                    f"doc={ex.doc_id} word={ex.surface_word} "
                    f"min_prefix_rank={min(ranks_prefix) if ranks_prefix else None} "
                    f"max_div={max(divs) if divs else None:.6f}",
                    flush=True,
                )
                del out
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[rate-complete] {model_key} domain={domain_key} rate={rate:.1f} examples={done_examples}", flush=True)
            if args.smoke:
                break
        if args.smoke:
            break

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rows_jsonl.exists():
        df = pd.read_json(rows_jsonl, lines=True)
        csv_path = output_dir / f"probe_results_v2_{model_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[saved-model] {csv_path} rows={len(df)}", flush=True)


def make_summary(df: pd.DataFrame, output_dir: Path):
    rows = []
    for (model, model_key, domain, domain_key, rate), g in df.groupby(["model", "model_key", "domain", "domain_key", "word_fragmentation_rate"]):
        med = g.groupby("layer")[["recovery_rank_prefix", "recovery_prob_prefix", "recovery_rank_last", "attn_entropy", "hidden_state_divergence"]].median()
        best_rank_layer = int(med["recovery_rank_prefix"].idxmin())
        best_prob_layer = int(med["recovery_prob_prefix"].idxmax())
        peak_div_layer = int(med["hidden_state_divergence"].idxmax())
        med_no_embed = med[med.index > 0]
        peak_div_no_embed_layer = int(med_no_embed["hidden_state_divergence"].idxmax()) if not med_no_embed.empty else None
        entropy = med["attn_entropy"].dropna()
        peak_entropy_layer = int(entropy.idxmax()) if not entropy.empty else None
        rows.append(
            {
                "model": model,
                "model_key": model_key,
                "domain": domain,
                "domain_key": domain_key,
                "word_fragmentation_rate": rate,
                "n_examples": int(g[["example_id"]].drop_duplicates().shape[0]),
                "min_recovery_rank_prefix": float(med["recovery_rank_prefix"].min()),
                "min_rank_layer": best_rank_layer,
                "max_recovery_prob_prefix": float(med["recovery_prob_prefix"].max()),
                "max_prob_layer": best_prob_layer,
                "min_recovery_rank_last": float(med["recovery_rank_last"].min()),
                "peak_attn_entropy": float(entropy.max()) if not entropy.empty else np.nan,
                "peak_entropy_layer": peak_entropy_layer,
                "peak_hidden_divergence": float(med["hidden_state_divergence"].max()),
                "peak_divergence_layer": peak_div_layer,
                "peak_hidden_divergence_no_embed": float(med_no_embed["hidden_state_divergence"].max()) if not med_no_embed.empty else np.nan,
                "peak_divergence_layer_no_embed": peak_div_no_embed_layer,
                "median_hidden_divergence": float(g["hidden_state_divergence"].median()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["model_key", "domain_key", "word_fragmentation_rate"])
    summary.to_csv(output_dir / "probe_summary_v2.csv", index=False)

    integrity_rows = []
    for (model, model_key, domain, domain_key), g in df.groupby(["model", "model_key", "domain", "domain_key"]):
        integrity_rows.append(
            {
                "model": model,
                "model_key": model_key,
                "domain": domain,
                "domain_key": domain_key,
                "rates": int(g["word_fragmentation_rate"].nunique()),
                "examples": int(g[["word_fragmentation_rate", "example_id"]].drop_duplicates().shape[0]),
                "rows": int(len(g)),
                "final_norm_path": ";".join(sorted(str(x) for x in g["final_norm_path"].dropna().unique())),
                "attention_rows": int(g["attn_entropy"].notna().sum()),
                "negative_entropy_rows": int((g["attn_entropy"].dropna() < 0).sum()),
                "divergence_min": float(g["hidden_state_divergence"].min()),
                "divergence_max": float(g["hidden_state_divergence"].max()),
            }
        )
    pd.DataFrame(integrity_rows).to_csv(output_dir / "probe_integrity_v2.csv", index=False)
    print(f"[saved] {output_dir / 'probe_summary_v2.csv'}", flush=True)


def make_plots(df: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    model_keys = list(df["model_key"].drop_duplicates())
    domains = list(df["domain"].drop_duplicates())
    if not model_keys:
        return
    fig, axes = plt.subplots(3, len(model_keys), figsize=(5.2 * len(model_keys), 11), squeeze=False)
    for col, model_key in enumerate(model_keys):
        sub_m = df[df["model_key"] == model_key]
        model_name = sub_m["model"].iloc[0]

        ax = axes[0][col]
        for domain in domains:
            sub = sub_m[sub_m["domain"] == domain]
            if sub.empty:
                continue
            best = sub.groupby(["word_fragmentation_rate", "layer"])["recovery_rank_prefix"].median()
            best = best.groupby("word_fragmentation_rate").min()
            ax.plot(best.index, best.values + 1, marker="o", label=domain)
        ax.set_yscale("log")
        ax.set_xlabel("WFR")
        ax.set_ylabel("Best-layer median prefix rank + 1")
        ax.set_title(f"{model_name}: word recovery")
        ax.legend(fontsize=7)

        ax = axes[1][col]
        for domain in domains:
            sub = sub_m[sub_m["domain"] == domain].dropna(subset=["attn_entropy"])
            if sub.empty:
                continue
            peak = sub.groupby(["word_fragmentation_rate", "layer"])["attn_entropy"].mean()
            peak = peak.groupby("word_fragmentation_rate").max()
            ax.plot(peak.index, peak.values, marker="o", label=domain)
        ax.set_xlabel("WFR")
        ax.set_ylabel("Peak-layer attention entropy")
        ax.set_title(f"{model_name}: attention entropy")
        ax.legend(fontsize=7)

        ax = axes[2][col]
        for domain in domains:
            sub = sub_m[sub_m["domain"] == domain]
            if sub.empty:
                continue
            sub = sub[sub["layer"] > 0]
            peak = sub.groupby(["word_fragmentation_rate", "layer"])["hidden_state_divergence"].median()
            peak = peak.groupby("word_fragmentation_rate").max()
            ax.plot(peak.index, peak.values, marker="o", label=domain)
        ax.set_xlabel("WFR")
        ax.set_ylabel("Peak hidden-state divergence\n(layers >= 1)")
        ax.set_title(f"{model_name}: divergence")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "probe_plots_v2.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {output_dir / 'probe_plots_v2.png'}", flush=True)


def summarize_outputs(output_dir: Path):
    frames = []
    for path in sorted(output_dir.glob("probe_results_v2_*.jsonl")):
        if path.stat().st_size == 0:
            continue
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        print(f"[summarize] no per-model JSONL files found in {output_dir}", flush=True)
        return
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(output_dir / "probe_results_v2.csv", index=False)
    make_summary(full, output_dir)
    make_plots(full, output_dir)
    print(f"[saved] {output_dir / 'probe_results_v2.csv'} rows={len(full)}", flush=True)


def parse_csv_values(text: str, valid: Iterable[str] | None = None) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if valid is not None:
        valid_set = set(valid)
        bad = [x for x in values if x not in valid_set]
        if bad:
            raise ValueError(f"Unknown values: {bad}; valid={sorted(valid_set)}")
    return values


def parse_rates(text: str) -> list[float]:
    rates = [round(float(x.strip()), 1) for x in text.split(",") if x.strip()]
    for rate in rates:
        if rate not in WFR_LEVELS:
            raise ValueError(f"Unsupported WFR rate {rate}; valid={WFR_LEVELS}")
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODEL_SPECS.keys()))
    parser.add_argument("--domains", default=",".join(DOMAINS.keys()))
    parser.add_argument("--rates", default=",".join(str(x) for x in WFR_LEVELS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--n-docs-override", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    args = parser.parse_args()

    if os.environ.get("HF_ALLOW_DOWNLOAD", "0") != "1":
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize_outputs(output_dir)
        return

    model_keys = parse_csv_values(args.models, MODEL_SPECS.keys())
    domain_keys = parse_csv_values(args.domains, DOMAINS.keys())
    rates = parse_rates(args.rates)
    if args.smoke:
        model_keys = model_keys[:1]
        domain_keys = domain_keys[:1]
        rates = rates[:1]

    with (output_dir / "probe_v2_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "models": model_keys,
                "domains": domain_keys,
                "rates": rates,
                "max_context_chars": args.max_context_chars,
                "n_docs_override": args.n_docs_override,
                "smoke": args.smoke,
                "model_specs": {k: asdict(MODEL_SPECS[k]) for k in model_keys},
                "domain_specs": {k: asdict(DOMAINS[k]) for k in domain_keys},
                "target_words_by_domain": {k: TARGET_WORDS_BY_DOMAIN[k] for k in domain_keys},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        run_for_model(model_key, spec, domain_keys, rates, output_dir, args)

    if not args.no_summarize:
        summarize_outputs(output_dir)


if __name__ == "__main__":
    main()
