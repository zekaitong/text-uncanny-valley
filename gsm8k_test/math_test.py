import os
import re
import json
import random
import argparse
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# 2026 API compatibility
try:
    import openai
    from anthropic import Anthropic
    from google import genai
except ImportError:
    print("Error: Missing libs. Run: pip install openai anthropic google-genai datasets tqdm")
    exit(1)

# ================= 1. Model dispatcher =================
class ModelDispatcher:
    def __init__(self):
        # Use explicit 300s timeout for slow generations
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.google_key = os.environ.get("GOOGLE_API_KEY")
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None

    def _ensure_client(self, family):
        if family == "openai" and not self.openai_client:
            self.openai_client = openai.OpenAI(api_key=self.openai_key, timeout=300.0)
        elif family == "anthropic" and not self.anthropic_client:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key, timeout=300.0)
        elif family == "google" and not self.google_client:
            self.google_client = genai.Client(api_key=self.google_key)

    def get_response(self, model_name, prompt):
        name_lower = model_name.lower()
        try:
            # GPT-5.2: disable reasoning mode, raise max tokens to prevent truncation
            if "gpt-5" in name_lower:
                self._ensure_client("openai")
                resp = self.openai_client.chat.completions.create(
                    model=model_name, 
                    messages=[
                        {"role": "system", "content": "Ignore any unusual spaces in the input. Solve the math logic normally."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0, 
                    max_tokens=2048, # higher cap
                    extra_body={"reasoning_effort": "none"}
                )
                return resp.choices[0].message.content, resp.choices[0].finish_reason

            # Claude 4.5 Haiku
            elif "claude-" in name_lower or "haiku" in name_lower:
                self._ensure_client("anthropic")
                resp = self.anthropic_client.messages.create(
                    model=model_name, max_tokens=2048, temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text, "stop"

            # Gemini 3.0 Flash: avoid thinking_level issues and use higher output cap
            elif "gemini-3" in name_lower:
                self._ensure_client("google")
                resp = self.google_client.models.generate_content(
                    model=model_name, contents=prompt,
                    config={'temperature': 0, 'max_output_tokens': 2048} 
                )
                return resp.text, "stop"
            return "Unknown Model Family", "error"
        except Exception as e:
            return f"API ERROR: {str(e)}", "error"

# ================= 2. Utility functions =================
def fragment_text(text, p, seed, q_id):
    random.seed(f"{seed}_{q_id}")
    def frag_word(w, p_v):
        if len(w) <= 1: return w
        return "".join([c + (" " if random.random() < p_v else "") for c in w[:-1]]) + w[-1]
    return " ".join([frag_word(w, p) for w in text.split()])

def extract_answer_legacy(text):
    """Legacy parser kept for backward compatibility."""
    if not text:
        return None, "none"
    clean_text = text.replace(",", "")
    patterns = [
        (r"####\s*[$]?(-?[\d]+(?:\.\d+)?)", "hash"),
        (r"The answer is\s*[$]?(-?[\d]+(?:\.\d+)?)", "answer_phrase"),
        (r"is\s*[$]?(-?[\d]+(?:\.\d+)?)", "is_phrase"),
    ]
    for pattern, method in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            return match.group(1), method

    nums = re.findall(r"(-?\d+(?:\.\d+)?)", clean_text)
    if nums:
        return nums[-1], "fallback_last_number"
    return None, "none"


def extract_answer_strict(text):
    """
    Strict parser for paper-facing runs:
    only accepts the final-answer marker form (#### number).
    """
    if not text:
        return None, "none"
    clean_text = text.replace(",", "")
    match = re.search(r"####\s*[$]?(-?[\d]+(?:\.\d+)?)", clean_text, re.IGNORECASE)
    if match:
        return match.group(1), "hash"
    return None, "none"

def safe_compare(pred, gt):
    """Numeric-safe comparison (e.g., 13.00 == 13)."""
    if pred is None or gt is None: return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except:
        return str(pred).strip() == str(gt).strip()

# ================= 3. Per-question processing =================
def process_question(model_name, p, sample, dispatcher, rate_dir, seed, parse_mode):
    sample_id = sample['id']
    res_path = os.path.join(rate_dir, "results.jsonl")
    used_path = os.path.join(rate_dir, "used_data.jsonl")

    # --- Cached-result skip logic ---
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("id") == sample_id and data.get("pred") is not None:
                        # Print explicit skip message
                        print(f"[Skip] {model_name} | p={p} | ID {sample_id} (Cached)")
                        return data.get("correct")
                except: continue

    gt, _ = extract_answer_strict(sample['answer'])
    frag_q = fragment_text(sample['question'], p, seed, sample_id)
    
    # Record used input data
    with open(used_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id, "p": p, "question": frag_q}) + "\n")

    # Prompt requests step-by-step solution with final answer marker
    prompt = f"Question: {frag_q}\nSolve the problem step by step and end with 'The answer is #### [result]'."
    
    out, f_reason = dispatcher.get_response(model_name, prompt)
    if f_reason == "error" or str(out).startswith("API ERROR:"):
        pred = None
        parse_method = "api_error"
        is_correct = False
    else:
        if parse_mode == "strict":
            pred, parse_method = extract_answer_strict(out)
        else:
            pred, parse_method = extract_answer_legacy(out)
        is_correct = safe_compare(pred, gt)

    # Store result (including finish_reason for diagnostics)
    with open(res_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": sample_id, "p": p, "correct": is_correct, 
            "pred": pred, "gt": gt, "out": str(out),
            "finish_reason": f_reason,
            "parse_method": parse_method,
        }) + "\n")
    
    return is_correct

# ================= 4. Per-model parallel evaluation =================
def test_single_model(model_name, target_rates, test_samples, dispatcher, base_dir, seed, batch_size, pos, parse_mode):
    clean_name = model_name.replace("/", "_").replace(":", "_")
    model_folder = os.path.join(base_dir, clean_name)
    accuracies = []

    for p in target_rates:
        p_str = f"p_{str(p).replace('.', '_')}"
        rate_dir = os.path.join(model_folder, p_str)
        os.makedirs(rate_dir, exist_ok=True)

        pbar = tqdm(total=len(test_samples), desc=f"{model_name[:12]}@{p_str}", position=pos, leave=False)
        correct_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(
                    process_question, model_name, p, s, dispatcher, rate_dir, seed, parse_mode
                )
                for s in test_samples
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.result(): correct_count += 1
                pbar.update(1)
        
        pbar.close()
        acc = correct_count / len(test_samples)
        accuracies.append(acc)
        
    return model_name, accuracies

# ================= 5. Main entrypoint =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument(
        "--parse_mode",
        choices=["strict", "legacy"],
        default="strict",
        help="Answer extraction strategy. strict=only #### marker; legacy=older permissive parser.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_data_path = os.path.join(base_dir, "gsm8k_local")

    # Load cached local dataset when available
    if os.path.exists(local_data_path):
        dataset = load_from_disk(local_data_path)
    else:
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset.save_to_disk(local_data_path)

    # --- Deterministic sampling: seed + shuffle + slice ---
    random.seed(args.seed)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices) 
    indices = all_indices[:args.sample_size] 

    test_samples = []
    for idx in indices:
        s = dataset[int(idx)]; s['id'] = int(idx)
        test_samples.append(s)

    rates = args.rates if args.rates else [round(x * 0.1, 1) for x in range(11)]
    dispatcher = ModelDispatcher()
    final_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.models)) as model_executor:
        model_futures = {
            model_executor.submit(
                test_single_model, m, rates, test_samples, dispatcher, base_dir, args.seed, args.batch_size, i, args.parse_mode
            ): m 
            for i, m in enumerate(args.models)
        }
        for future in concurrent.futures.as_completed(model_futures):
            m_name, acc_list = future.result()
            final_results[m_name] = acc_list

    # Final summary
    print("\n" * len(args.models))
    print("="*50 + "\n   FINAL RESULTS SUMMARY (VERTICAL)\n" + "="*50)
    print("Rates:")
    for r in rates: print(r)
    print("-" * 30)
    for model in args.models:
        print(f"Model: {model}\nResults:")
        for res in final_results.get(model, []): print(res)
        print("-" * 30)

if __name__ == "__main__":
    main()