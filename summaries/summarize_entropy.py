import argparse
import csv
import json
import os
import re
from collections import Counter

import numpy as np
import tiktoken
from tqdm import tqdm


# ─── 1. Load tokenizer ────────────────────────────────────────

gpt_tokenizer = tiktoken.encoding_for_model("gpt-4")
claude_tokenizer = tiktoken.get_encoding("cl100k_base")

tokenizers = {
    'gpt':    ('tiktoken', gpt_tokenizer),
    'claude': ('tiktoken', claude_tokenizer),
}

# ─── 2. tokenize ──────────────────────────────────────────────

def tokenize(text, tokenizer):
    ids = tokenizer.encode(text)
    return ids

# ─── 3. Compute token entropy ─────────────────────────────────

def compute_entropy(text, tokenizer):
    ids = tokenize(text, tokenizer)
    if not ids:
        return 0.0
    counts = Counter(ids)
    total  = sum(counts.values())
    probs  = np.array([c / total for c in counts.values()])
    return -np.sum(probs * np.log2(probs))

# ─── 4. Read used_data.jsonl ──────────────────────────────────

def load_original_contexts(folder_path):
    jsonl_path = os.path.join(folder_path, 'used_data.jsonl')
    contexts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'original_context' in obj:
                contexts.append(obj['original_context'])
    return contexts

# ─── 5. Sort folders: ori first, then w1..w10 ────────────────

def sort_subfolders(folders):
    def sort_key(name):
        if name == 'ori':
            return (0, 0, '')
        # Extract trailing number regardless of prefix
        match = re.search(r'(\d+)$', name)
        if match:
            return (1, int(match.group(1)), name)
        return (2, 0, name)
    return sorted(folders, key=sort_key)

# ─── 6. Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute tokenization entropy across rate subfolders."
    )
    parser.add_argument("base_dir")
    parser.add_argument(
        "--tokenizer",
        choices=["gpt", "claude", "all"],
        default="all",
    )
    parser.add_argument("--json_out", type=str, help="Optional JSON output path")
    parser.add_argument("--csv_out", type=str, help="Optional CSV output path")
    args = parser.parse_args()

    base_dir = args.base_dir
    model_arg = args.tokenizer

    if model_arg == 'all':
        selected = tokenizers
    elif model_arg in tokenizers:
        selected = {model_arg: tokenizers[model_arg]}
    else:
        raise ValueError(f"Unknown tokenizer: {model_arg}")

    # Sorted folder list
    subfolders = sort_subfolders([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.exists(os.path.join(base_dir, d, 'used_data.jsonl'))
    ])

    # Print folder list for quick inspection
    print("\n=== Subfolders ===")
    for name in subfolders:
        print(name)
    print()

    outputs = {}
    for model_name, (_, tok) in selected.items():
        results = []

        # Progress stage: compute first, print after
        for name in tqdm(subfolders, desc=model_name):
            folder_path = os.path.join(base_dir, name)
            contexts = load_original_contexts(folder_path)

            if not contexts:
                results.append(None)
                continue

            entropies = [compute_entropy(ctx, tok) for ctx in contexts]
            results.append(np.mean(entropies))

        outputs[model_name] = results
        print(f"\n=== Tokenization Entropy ({model_name}) ===")
        for val in results:
            if val is None:
                print("[no data]")
            else:
                print(f"{val:.4f}")
        print()

    # If only one tokenizer selected, emit a flat format for plotting convenience
    if len(outputs) == 1:
        tok_name = list(outputs.keys())[0]
        if args.json_out:
            payload = {
                "tokenizer": tok_name,
                "folders": subfolders,
                "entropy": outputs[tok_name],
            }
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        if args.csv_out:
            os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["folder", "entropy"])
                for folder, ent in zip(subfolders, outputs[tok_name]):
                    writer.writerow([folder, ent])
    else:
        if args.json_out:
            payload = {"folders": subfolders, "entropies": outputs}
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
