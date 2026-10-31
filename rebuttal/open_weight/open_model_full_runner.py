#!/usr/bin/env python3
"""Full local open-model runner for AbsenceBench, NIAH, and GSM8K math.

One Slurm job should run one model. The model is loaded once, every sample is
appended to JSONL immediately, and completed ok rows are skipped on resume.
"""

import argparse
import gc
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
# This release includes the exact evaluator snapshot used by the rebuttal runs.
TESTS_DIR = Path(__file__).resolve().parent / "legacy_evaluator"
from legacy_evaluator.scorer import evaluate_response, load_documents


MODEL_PATHS = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-awq-int4": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

MODEL_REVISIONS = {
    "qwen2.5-7b-instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "qwen3-30b-a3b": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "llama-3.1-8b-instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "llama-3.1-70b-awq-int4": "2123003760781134cfc31124aa6560a45b491fdf",
    "gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

RATES = [("ori", 0.0)] + [(str(i), i / 10.0) for i in range(1, 11)]

ABSENCE_SOURCES = {
    "ld": ("legal_documents", REPO_ROOT / "data/results/ld/claude-haiku-4-5/wa"),
    "gp": ("github_prs", REPO_ROOT / "data/results/gp/claude-haiku-4-5/wa"),
    "gsm": ("math_reasoning", REPO_ROOT / "data/results/gsm/claude-haiku-4-5/wa"),
}

NIAH_SOURCES = {
    "ld": ("legal_documents", REPO_ROOT / "data/results/NIAH/ld/gpt-5.2/wa_bn"),
    "gp": ("github_prs", REPO_ROOT / "data/results/NIAH/gp/gpt-5.2/wa_bn"),
    "gsm": ("math_reasoning", REPO_ROOT / "data/results/NIAH/gsm/gpt-5.2/wa_bn"),
}

GSM8K_LOCAL = REPO_ROOT / "gsm8k_test/gsm8k_local"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_ids(path):
    completed = set()
    if not path.exists():
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok":
                completed.add(str(item.get("id")))
    return completed


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_summary(task_dir, model_key, task_name, results_path):
    raw_rows = []
    by_id = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    pass
                else:
                    raw_rows.append(row)
                    row_id = str(row.get("id", f"__row_{line_no}"))
                    current = by_id.get(row_id)
                    if current is None:
                        by_id[row_id] = row
                    elif current.get("status") != "ok" and row.get("status") == "ok":
                        by_id[row_id] = row
                    elif current.get("status") == row.get("status"):
                        by_id[row_id] = row
    rows = list(by_id.values())
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    f1_rows = [r for r in ok_rows if r.get("micro_f1") is not None]
    correct_rows = [r for r in ok_rows if r.get("correct") is not None]
    summary = {
        "model_key": model_key,
        "task": task_name,
        "raw_rows": len(raw_rows),
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": len(rows) - len(ok_rows),
        "avg_elapsed_sec": (
            sum(float(r.get("elapsed_sec", 0.0)) for r in ok_rows) / len(ok_rows)
            if ok_rows
            else None
        ),
        "avg_micro_f1": (
            sum(float(r.get("micro_f1", 0.0)) for r in f1_rows) / len(f1_rows)
            if f1_rows
            else None
        ),
        "avg_correct": (
            sum(1.0 if r.get("correct") else 0.0 for r in correct_rows) / len(correct_rows)
            if correct_rows
            else None
        ),
        "updated_at": utc_now(),
    }
    write_json(task_dir / "summary.json", summary)


def write_progress(out_base, model_key, payload):
    payload = dict(payload)
    payload["model_key"] = model_key
    payload["updated_at"] = utc_now()
    write_json(out_base / model_key / "progress.json", payload)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_items(items, shard_index, num_shards):
    if num_shards <= 1:
        return list(items)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return [item for pos, item in enumerate(items) if pos % num_shards == shard_index]


def format_chat(tokenizer, system_prompt, user_message):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\n{user_message}"
    return user_message


def generate_text(model, tokenizer, system_prompt, user_message, max_new_tokens):
    text = format_chat(tokenizer, system_prompt, user_message)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_tokens = int(inputs["input_ids"].shape[-1])
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    output_tokens = int(new_tokens.shape[-1])
    del inputs, out, new_tokens
    return response, input_tokens, output_tokens


def absence_prompt(document, document_type, use_needle):
    original = document["original_context"]
    modified = document.get("_modified_context") or document.get("modified_context", "")
    if use_needle:
        system = (
            f"You are helping someone identify components that were added to a {document_type}.\n"
            f"The person will provide their version of the {document_type}, but they may have added some random lines that were not in the original text.\n"
            f"Your task is to identify exactly which lines are not in the original {document_type}.\n"
            "List only the extra lines, nothing else."
        )
        user = (
            f"Here is the complete original {document_type}:\n"
            f"{original}\n\n"
            f"Now, here is my recitation with some extra lines that were not in the original {document_type}:\n\n"
            f"{modified}\n\n"
            f"What lines did I add to the {document_type}? Please list only the extra lines, nothing else."
        )
    else:
        system = (
            f"You are helping someone identify missing components from a {document_type}.\n"
            f"The person will provide their version of the {document_type}, but they may have missed some lines.\n"
            "Your task is to identify exactly which lines are missing from their version.\n"
            "List only the missing lines, nothing else."
        )
        user = (
            f"Here is the complete original {document_type}:\n\n"
            f"{original}\n\n"
            f"Now, here is my recitation which may be missing some lines:\n\n"
            f"{modified}\n\n"
            "What lines did I miss? Please list only the missing lines, nothing else."
        )
    return system, user


def run_absence_matrix(
    model,
    tokenizer,
    model_key,
    out_base,
    task_group,
    sources,
    use_needle,
    sample_size,
    seed,
    selected_domains=None,
    selected_rates=None,
    max_new_tokens=512,
    shard_index=0,
    num_shards=1,
):
    task_name = "niah" if use_needle else "absencebench"
    variant = "wa_bn" if use_needle else "wa"
    selected_domains = set(selected_domains) if selected_domains else set(sources)
    selected_rates = set(selected_rates) if selected_rates else {label for label, _ in RATES}
    for domain, (document_type, base_dir) in sources.items():
        if domain not in selected_domains:
            continue
        for rate_label, rate_value in RATES:
            if rate_label not in selected_rates:
                continue
            input_path = base_dir / rate_label / "processed_data.jsonl"
            task_dir = out_base / model_key / task_name / domain / variant / rate_label
            results_path = task_dir / "results.jsonl"
            completed = load_completed_ids(results_path)
            docs = load_documents(str(input_path), sample_size=sample_size, random_seed=seed)
            docs = shard_items(list(enumerate(docs)), shard_index, num_shards)
            write_progress(
                out_base,
                model_key,
                {
                    "phase": "running",
                    "task": task_name,
                    "domain": domain,
                    "variant": variant,
                    "rate_label": rate_label,
                    "rate": rate_value,
                    "completed_ok": len(completed),
                    "target": len(docs),
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                },
            )
            for idx, doc in docs:
                doc_id = str(doc.get("_id") if doc.get("_id") is not None else doc.get("id", idx))
                if doc_id in completed:
                    continue
                started = time.time()
                try:
                    system, user = absence_prompt(doc, document_type, use_needle=use_needle)
                    response, input_tokens, output_tokens = generate_text(
                        model,
                        tokenizer,
                        system,
                        user,
                        max_new_tokens=max_new_tokens,
                    )
                    evaluation = evaluate_response([response, 0, ""], doc, use_needle=use_needle, delimiter="\n")
                    row = {
                        "status": "ok",
                        "id": doc_id,
                        "model_key": model_key,
                        "task": task_name,
                        "domain": domain,
                        "variant": variant,
                        "rate_label": rate_label,
                        "rate": rate_value,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "elapsed_sec": round(time.time() - started, 3),
                        "micro_f1": evaluation.get("micro_f1"),
                        "evaluation": evaluation,
                        "model_response": response,
                        "created_at": utc_now(),
                    }
                    completed.add(doc_id)
                except Exception as exc:
                    row = {
                        "status": "error",
                        "id": doc_id,
                        "model_key": model_key,
                        "task": task_name,
                        "domain": domain,
                        "variant": variant,
                        "rate_label": rate_label,
                        "rate": rate_value,
                        "elapsed_sec": round(time.time() - started, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "created_at": utc_now(),
                    }
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                append_jsonl(results_path, row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write_summary(task_dir, model_key, task_name, results_path)
                write_progress(
                    out_base,
                    model_key,
                    {
                        "phase": "running",
                        "task": task_name,
                        "domain": domain,
                        "variant": variant,
                        "rate_label": rate_label,
                        "rate": rate_value,
                        "completed_ok": len(completed),
                        "target": len(docs),
                        "last_id": doc_id,
                        "last_status": row["status"],
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                    },
                )
                print(f"[done] {model_key} {task_name}/{domain}/{variant}/{rate_label} {doc_id} {row['status']}", flush=True)
            write_summary(task_dir, model_key, task_name, results_path)


def fragment_text(text, p, seed, q_id):
    random.seed(f"{seed}_{q_id}")

    def frag_word(word, p_value):
        if len(word) <= 1:
            return word
        return "".join(char + (" " if random.random() < p_value else "") for char in word[:-1]) + word[-1]

    return " ".join(frag_word(word, p) for word in text.split())


def extract_answer(text):
    if not text:
        return None
    clean = text.replace(",", "")
    for pattern in [
        r"####\s*[$]?(-?[\d]+(?:\.\d+)?)",
        r"The answer is\s*[$]?(-?[\d]+(?:\.\d+)?)",
        r"is\s*[$]?(-?[\d]+(?:\.\d+)?)",
    ]:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            return match.group(1)
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", clean)
    return nums[-1] if nums else None


def safe_compare(pred, gt):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except Exception:
        return str(pred).strip() == str(gt).strip()


def run_math_matrix(
    model,
    tokenizer,
    model_key,
    out_base,
    sample_size,
    seed,
    selected_rates=None,
    max_new_tokens=1024,
    shard_index=0,
    num_shards=1,
):
    dataset = load_from_disk(str(GSM8K_LOCAL))
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:sample_size]
    indices = shard_items(indices, shard_index, num_shards)
    task_name = "math_gsm8k"
    selected_rates = set(selected_rates) if selected_rates else {label for label, _ in RATES}
    for rate_label, rate_value in RATES:
        if rate_label not in selected_rates:
            continue
        p_label = f"p_{str(rate_value).replace('.', '_')}"
        task_dir = out_base / model_key / task_name / p_label
        results_path = task_dir / "results.jsonl"
        completed = load_completed_ids(results_path)
        write_progress(
            out_base,
            model_key,
            {
                "phase": "running",
                "task": task_name,
                "rate_label": rate_label,
                "rate": rate_value,
                "completed_ok": len(completed),
                "target": len(indices),
                "shard_index": shard_index,
                "num_shards": num_shards,
            },
        )
        for idx in indices:
            sample = dataset[int(idx)]
            sample_id = str(int(idx))
            if sample_id in completed:
                continue
            gt = extract_answer(sample["answer"])
            frag_q = fragment_text(sample["question"], rate_value, seed, sample_id)
            system = "Ignore any unusual spaces in the input. Solve the math logic normally."
            user = f"Question: {frag_q}\nSolve the problem step by step and end with 'The answer is #### [result]'."
            started = time.time()
            try:
                response, input_tokens, output_tokens = generate_text(model, tokenizer, system, user, max_new_tokens=max_new_tokens)
                pred = extract_answer(response)
                correct = safe_compare(pred, gt)
                row = {
                    "status": "ok",
                    "id": sample_id,
                    "model_key": model_key,
                    "task": task_name,
                    "rate_label": rate_label,
                    "rate": rate_value,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "elapsed_sec": round(time.time() - started, 3),
                    "pred": pred,
                    "gt": gt,
                    "correct": correct,
                    "model_response": response,
                    "created_at": utc_now(),
                }
                completed.add(sample_id)
            except Exception as exc:
                row = {
                    "status": "error",
                    "id": sample_id,
                    "model_key": model_key,
                    "task": task_name,
                    "rate_label": rate_label,
                    "rate": rate_value,
                    "elapsed_sec": round(time.time() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now(),
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            append_jsonl(results_path, row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_summary(task_dir, model_key, task_name, results_path)
            write_progress(
                out_base,
                model_key,
                {
                    "phase": "running",
                    "task": task_name,
                    "rate_label": rate_label,
                    "rate": rate_value,
                    "completed_ok": len(completed),
                    "target": len(indices),
                    "last_id": sample_id,
                    "last_status": row["status"],
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                },
            )
            print(f"[done] {model_key} {task_name}/{p_label} {sample_id} {row['status']}", flush=True)
        write_summary(task_dir, model_key, task_name, results_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_PATHS))
    parser.add_argument("--model-path", help="Optional local model directory; overrides the pinned Hugging Face checkpoint.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads instead of requiring a populated local cache.")
    parser.add_argument("--absence-sample-size", type=int, default=100)
    parser.add_argument("--math-sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default=str(REPO_ROOT / "data/results/open_model_full"))
    parser.add_argument("--tasks", default="absencebench,niah,math_gsm8k")
    parser.add_argument("--absence-domains", default="ld,gp,gsm")
    parser.add_argument("--niah-domains", default="ld,gp,gsm")
    parser.add_argument("--rates", default="", help="Comma-separated rate labels to run for AbsenceBench/NIAH, e.g. ori,1,5,10")
    parser.add_argument("--math-rates", default="", help="Comma-separated rate labels to run for math_gsm8k; defaults to --rates when set")
    parser.add_argument("--absence-max-new-tokens", type=int, default=512)
    parser.add_argument("--math-max-new-tokens", type=int, default=1024)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    tasks = set(parse_csv(args.tasks))
    absence_domains = parse_csv(args.absence_domains)
    niah_domains = parse_csv(args.niah_domains)
    selected_rates = parse_csv(args.rates)
    selected_math_rates = parse_csv(args.math_rates) or selected_rates
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    if not args.allow_download:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        from transformers.utils import logging

        logging.disable_progress_bar()
        logging.set_verbosity_error()
    except Exception:
        pass

    out_base = Path(args.output_base)
    write_progress(out_base, args.model_key, {"phase": "starting"})
    print(f"[start] model={args.model_key} host={os.uname().nodename}", flush=True)
    print(f"[env] cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] device={torch.cuda.get_device_name(0)}", flush=True)

    path = args.model_path or MODEL_PATHS[args.model_key]
    revision = None if args.model_path else MODEL_REVISIONS[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        revision=revision,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "revision": revision,
        "local_files_only": not args.allow_download,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.model_key == "llama-3.1-70b-awq-int4" else torch.bfloat16,
        "use_safetensors": True,
    }
    if os.environ.get("ATTN_IMPLEMENTATION"):
        model_kwargs["attn_implementation"] = os.environ["ATTN_IMPLEMENTATION"]
        print(f"[env] attn_implementation={model_kwargs['attn_implementation']}", flush=True)

    uses_device_map = False
    if args.model_key == "llama-3.1-70b-awq-int4":
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["low_cpu_mem_usage"] = True
        uses_device_map = True
    if args.model_key == "gemma-3-4b-it" and os.environ.get("GEMMA_CPU_VISION") == "1":
        model_kwargs["device_map"] = {
            "model.vision_tower": "cpu",
            "model.multi_modal_projector": "cpu",
            "model.language_model": 0,
            "lm_head": 0,
        }
        model_kwargs["low_cpu_mem_usage"] = True
        uses_device_map = True

    model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    if not uses_device_map:
        model.to("cuda")
    model.eval()

    if "absencebench" in tasks:
        run_absence_matrix(
            model,
            tokenizer,
            args.model_key,
            out_base,
            "absencebench",
            ABSENCE_SOURCES,
            False,
            args.absence_sample_size,
            args.seed,
            selected_domains=absence_domains,
            selected_rates=selected_rates,
            max_new_tokens=args.absence_max_new_tokens,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    if "math_gsm8k" in tasks or "math" in tasks:
        run_math_matrix(
            model,
            tokenizer,
            args.model_key,
            out_base,
            args.math_sample_size,
            args.seed,
            selected_rates=selected_math_rates,
            max_new_tokens=args.math_max_new_tokens,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    if "niah" in tasks:
        run_absence_matrix(
            model,
            tokenizer,
            args.model_key,
            out_base,
            "niah",
            NIAH_SOURCES,
            True,
            args.absence_sample_size,
            args.seed,
            selected_domains=niah_domains,
            selected_rates=selected_rates,
            max_new_tokens=args.absence_max_new_tokens,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_progress(out_base, args.model_key, {"phase": "finished"})
    print(f"[finish] model={args.model_key}", flush=True)


if __name__ == "__main__":
    main()
