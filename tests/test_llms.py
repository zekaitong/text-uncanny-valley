"""
This script tests LLMs on their ability to recall the omissions made in a document.
It accepts as input a processed JSON Lines file (e.g. one produced by process_data.py)
containing the following keys for each document:
  - "original_context": The full document
  - "_modified_context": The document after randomly deleting some lines.
  - "omitted_context": A list of the exact text of each changed line
    that was removed.

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed lines.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Example usage:
    python test_llms.py --input_file data/[document].jsonl
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output results/[document]_gpt-4.json] [--batch_size 5]
"""

import argparse
import concurrent.futures
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from llm_providers import LLMProvider
from tqdm import tqdm


def load_documents(
    jsonl_file: str, sample_size: int = None, random_seed: int = None
) -> List[Dict[str, Any]]:
    """
    Load documents from the given jsonl file, with optional sub-sampling
    """
    documents = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    if sample_size and sample_size < len(documents):
        if random_seed is not None:
            random.seed(random_seed)
        return random.sample(documents, sample_size)

    return documents


def evaluate_response(
    response_list: str,
    document_data: Dict[str, Any],
    use_needle: bool,
    delimiter: str = "\n",
) -> Dict[str, Any]:
    """
    Evaluate the model's response to determine if it correctly identified the omitted lines
    Reports micro f1 scores
    """
    original_lines = document_data["original_context"].split(delimiter)
    omitted_indices = document_data["omitted_index"]
    # if we are testing needle in a haystack, then we will not use omitted_indices for omissions
    if use_needle:
        needles = document_data["needles"]  # gives us list of used needles

    if response_list[0] == None:
        response = ""
        thinking_tokens = 0
        thinking_summary = ""
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]
        if thinking_tokens is None:
            thinking_tokens = 0
        thinking_summary = response_list[2] if len(response_list) > 2 else ""

    # some models might include thinking tokens in their response
    if "</think>" in response:
        response = response[response.index("</think>") + 8 :]

    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": [],
        "excess_identified_lines": [],
    }

    # Find repeat lines
    cleaned_line_counts = Counter(line.strip().lower() for line in original_lines)
    repeat_lines = [line for line, count in cleaned_line_counts.items() if count > 1]
    repeat_line_omitted_indices = {line.strip().lower(): [] for line in repeat_lines}

    for line_idx, line in enumerate(original_lines):
        # Clean up the line for comparison (remove punctuation, extra spaces, etc.)
        clean_line = line.strip().lower()

        if clean_line in repeat_lines:
            # we save omit idx
            # THIS IS WRONG IF WE ARE USING NEEDLE!! NO LINES ARE OMITTED EVEN IF WE ARE SAVING OMITTED IDX
            if line_idx in omitted_indices:
                repeat_line_omitted_indices[clean_line].append(line_idx)

    if not use_needle:
        repeat_line_omission_counts = {
            clean_line: len(val)
            for clean_line, val in repeat_line_omitted_indices.items()
        }
    else:
        # WE DO NOT OMIT ANY LINES WHEN USING NEEDLE
        # WE WILL ULTIMATELY PENALIZE IF THE MODEL GIVES **ANY** LINES FROM THE OG DOC IN ITS RESPONSE
        repeat_line_omission_counts = {}

    # Check if omitted counts align with the model's response
    response_lines_lower = [l.strip().lower() for l in response.split("\n")]
    for cleaned_duplicate in repeat_line_omission_counts:
        model_response_line_count = response_lines_lower.count(cleaned_duplicate)

        count_difference = (
            model_response_line_count - repeat_line_omission_counts[cleaned_duplicate]
        )

        # if the model gave too many of the duplicate line
        if count_difference > 0:
            results["fp"] += count_difference
            results["tp"] += repeat_line_omission_counts[cleaned_duplicate]
            for _ in range(repeat_line_omission_counts[cleaned_duplicate]):
                results["identified_lines"].append(cleaned_duplicate)
            for _ in range(count_difference):
                results["wrongly_identified_lines"].append(cleaned_duplicate)
        # if the model gave too few of the duplicate line
        elif count_difference < 0:
            results["fn"] += count_difference * -1
            results["tp"] += model_response_line_count
            for _ in range(count_difference * -1):
                results["unidentified_lines"].append(cleaned_duplicate)
        else:
            results["tp"] += model_response_line_count
            # equivalent to adding repeat_line_omission_counts[cleaned_duplicate]
            for _ in range(repeat_line_omission_counts[cleaned_duplicate]):
                results["identified_lines"].append(cleaned_duplicate)

    if use_needle:
        # NIAH scoring: strict line-level matching against inserted needles.
        response_lines = [line.strip() for line in response.split(delimiter) if line.strip()]
        response_lines_lower = [line.lower() for line in response_lines]
        original_lines_lower = set(line.strip().lower() for line in original_lines if line.strip())

        needle_counter = Counter(needle.strip().lower() for needle in needles if needle.strip())
        response_counter = Counter(response_lines_lower)

        matched_needles = Counter()
        for needle_line, needle_count in needle_counter.items():
            matched_count = min(needle_count, response_counter.get(needle_line, 0))
            if matched_count > 0:
                results["tp"] += matched_count
                matched_needles[needle_line] += matched_count
                for _ in range(matched_count):
                    results["identified_lines"].append(needle_line)

            missed_count = needle_count - matched_count
            if missed_count > 0:
                results["fn"] += missed_count
                for _ in range(missed_count):
                    results["unidentified_lines"].append(needle_line)

        # Any response line not consumed by true-needle matching is a false positive.
        for response_line, response_line_lower in zip(response_lines, response_lines_lower):
            if matched_needles.get(response_line_lower, 0) > 0:
                matched_needles[response_line_lower] -= 1
                continue

            results["fp"] += 1
            results["wrongly_identified_lines"].append(response_line)
            if response_line_lower not in original_lines_lower:
                results["excess_identified_lines"].append(response_line)
    else:
        for idx, line in enumerate(original_lines):
            # Clean up the line for comparison (remove punctuation, extra spaces, etc.)
            clean_line = line.strip().lower()
            if clean_line in repeat_line_omitted_indices:
                continue
            if clean_line and clean_line in response_lines_lower:
                if idx in omitted_indices:
                    results["tp"] += 1
                    results["identified_lines"].append(line)
                else:
                    results["fp"] += 1
                    results["wrongly_identified_lines"].append(line)
            elif clean_line and clean_line not in response_lines_lower:
                if idx in omitted_indices:
                    results["fn"] += 1
                    results["unidentified_lines"].append(line)

        response_lines = response.split(delimiter)
        original_lines_lower = [line.strip().lower() for line in original_lines]

        for response_line in response_lines:
            clean_response_line = response_line.strip().lower()
            # Skip empty lines
            if not clean_response_line:
                continue
            # Check if this response line appears in the original document
            if clean_response_line not in original_lines_lower:
                results["fp"] += 1
                results["excess_identified_lines"].append(response_line.strip())

    # Calculate micro_f1 score
    try:
        results["micro_f1"] = (
            2 * results["tp"] / (2 * results["tp"] + results["fp"] + results["fn"])
        )
    except Exception:
        results["micro_f1"] = 0

    # if len(omitted_indices) == 0:
    #     results["micro_f1"] = 1 - results["fp"] / len(original_lines)
    if not use_needle and len(omitted_indices) == 0:
        results["micro_f1"] = 1 - results["fp"] / len(original_lines)

    results["thinking_token"] = thinking_tokens
    results["thinking_summary"] = thinking_summary
    results["needle_depths"] = document_data.get("needle_depths", [])
    return results


def process_document(
    document: Dict[str, Any],
    document_idx: int,
    model_provider: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    thinking: bool,
    use_needle: bool,
    document_type: str,
    delimiter: str,
    reasoning_effort: str = None,
    icl_examples: str = None,
) -> Dict[str, Any]:
    """Process a single document with the specified model"""

    if not document_type:
        document_type = "document"

    def get_modified_context(document):
        """Handles backward compatibility for modified context field name"""
        return document.get("_modified_context") or document.get("modified_context", "")

    icl_header = f"Below are some examples of the task:\n\n{icl_examples}\n\n---\n\n" if icl_examples else ""

    if user_message:
        # Use custom user message with template substitution
        formatted_message = icl_header + user_message.format(
            document_type=document_type,
            original_context=document["original_context"],
            _modified_context=get_modified_context(document),
        )
    else:
        modified_ctx = get_modified_context(document)
        # Use default message
        formatted_message = icl_header + f"""Here is the complete original {document_type}:

{document["original_context"]}

Now, here is my recitation which may be missing some lines:

{modified_ctx}

What lines did I miss? Please list only the missing lines, nothing else."""
        
        if use_needle:
            formatted_message = f"""Here is the complete original {document_type}:
{document["original_context"]}

Now, here is my recitation with some extra lines that were not in the original {document_type}:

{modified_ctx}

What lines did I add to the {document_type}? Please list only the extra lines, nothing else."""
            
    try:
        # Get the appropriate provider
        provider = LLMProvider.get_provider(model_provider)

        # Get response from the provider
        # Only pass reasoning_effort to OpenAI provider for GPT-5 models
        if model_provider == "openai" and model_name.startswith("gpt-5"):
            response = provider.get_response(
                system_prompt, formatted_message, model_name, thinking, reasoning_effort
            )
        else:
            response = provider.get_response(
                system_prompt, formatted_message, model_name, thinking
            )

        evaluation = evaluate_response(
            response, document, use_needle, delimiter=delimiter
        )
        # Check for _id first (current format), then id (legacy format)
        doc_id = document.get("_id") if document.get("_id") is not None else document.get("id")
        if doc_id is None:
            raise ValueError(f"Document at index {document_idx} missing both '_id' and 'id' fields")
        evaluation["id"] = doc_id
        evaluation["model_response"] = response
        return evaluation

    except Exception as e:
        print(
            f"Error with {model_provider}/{model_name} on document {document_idx}: {str(e)}"
        )
        print(type(e))
        return None


def test_model(
    documents: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    system_prompt: str = None,
    user_message: str = None,
    batch_size: int = 5,
    thinking: bool = False,
    use_needle: bool = False,
    document_type: str = None,
    delimiter: str = "\n",
    reasoning_effort: str = None,
    icl_examples: str = None,
) -> Dict[str, Any]:
    """
    Test a model on all the documents and return the results, processing in batches
    """
    results = []
    total_documents = len(documents)

    if not document_type:
        document_type = "document"

    if not system_prompt:
        system_prompt = f"""You are helping someone identify missing components from a {document_type}. 
The person will provide their version of the {document_type}, but they may have missed some lines. 
Your task is to identify exactly which lines are missing from their version.
List only the missing lines, nothing else."""

    if use_needle:
        system_prompt = f"""You are helping someone identify components that were added to a {document_type}.
The person will provide their version of the {document_type}, but they may have added some random lines that were not in the original text.
Your task is to identify exactly which lines are not in the original {document_type}.
List only the extra lines, nothing else."""

    # Process docs in batches
    for batch_start in tqdm(range(0, total_documents, batch_size)):
        batch_end = min(batch_start + batch_size, total_documents)
        current_batch = documents[batch_start:batch_end]

        # Process the batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, document in enumerate(current_batch):
                document_idx = batch_start + i
                future = executor.submit(
                    process_document,
                    document,
                    document_idx,
                    model_provider,
                    model_name,
                    system_prompt,
                    user_message,
                    thinking,
                    use_needle,
                    document_type,
                    delimiter,
                    reasoning_effort,
                    icl_examples,
                )
                futures.append(future)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Add a small delay between batches to avoid rate limits
        if batch_end < total_documents:
            # print(f"Waiting a moment before the next batch...")
            time.sleep(2)

    # Calculate aggregate statistics
    accuracy_sum = sum(r["micro_f1"] for r in results)
    avg_accuracy = accuracy_sum / len(results) if results else 0
    thinking_sum = sum(r["thinking_token"] for r in results)
    avg_thinking = thinking_sum / len(results) if results else 0

    return {
        "model_provider": model_provider,
        "model_name": model_name,
        "total_documents": len(documents),
        "average_accuracy": avg_accuracy,
        "average_thinking_tokens": avg_thinking,
        "detailed_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test LLMs on their ability to identify omitted lines from documents"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the processed documents JSONL file",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Number of documents to sample for testing (default: use all)",
    )
    parser.add_argument(
        "--provider_models",
        type=str,
        nargs="+",
        default=["openai:gpt-4.1-nano-2025-04-14"],
        help='Provider and model pairs in the format "provider:model" '
        '(e.g., "openai:gpt-4 anthropic:claude-3-opus")',
    )
    parser.add_argument("--output", type=str, help="File name to save the test results")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of API calls to batch together (default: 5)",
    )
    parser.add_argument(
        "--use_needle", action="store_true", help="evalute with the NIAH setting"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Whether to enable the thinking mode or not",
    )
    parser.add_argument(
        "--document_type",
        type=str,
        help="The type of document you are evaluating (e.g. poem)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="The system prompt used in evaluation (if not provided, a generic prompt is used)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n",
        help="Delimiter to set for modified documents (default: newline)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for GPT-5 models. Only applies to GPT-5 models when set. Uses responses API when specified.",
    )
    parser.add_argument(
        "--user_message",
        type=str,
        help="Custom user message template. Use {document_type}, {original_context}, and {_modified_context} as placeholders",
    )
    parser.add_argument(
        "--icl_file",
        type=str,
        help="Path to a text file containing few-shot ICL examples",
    )

    args = parser.parse_args()

    # Check if input file exists
    documents_path = Path(args.input_file)
    if not documents_path.exists():
        print(f"Error: Documents file '{args.input_file}' does not exist!")
        return

    # Load ICL content if file is provided
    icl_content = None
    if args.icl_file:
        with open(args.icl_file, "r", encoding="utf-8") as f:
            icl_content = f.read()

    # Load and potentially sub-sample the documents
    documents = load_documents(args.input_file, args.sample_size, args.random_seed)
    if args.document_type:
        print(f"Loaded {len(documents)} {args.document_type.upper()} for testing")
    else:
        print(f"Loaded {len(documents)} documents for testing")

    # Automatically save sampled documents to used_data.jsonl
    if args.output:
        output_dir = Path(args.output).parent
        used_data_path = output_dir / "used_data.jsonl"
    else:
        used_data_path = Path("used_data.jsonl")

    with open(used_data_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved {len(documents)} sampled documents to {used_data_path}")

    # Parse provider:model pairs
    provider_models = []
    for pair in args.provider_models:
        if ":" not in pair:
            print(
                f"Warning: Skipping invalid provider-model pair '{pair}'. Format should be 'provider:model'"
            )
            continue
        provider, model = pair.split(":", 1)
        provider_models.append((provider, model))

    if not provider_models:
        print("Error: No valid provider-model pairs specified!")
        return

    # Initialize results dictionary
    all_results = {"test_date": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Test each provider-model pair
    for provider, model in provider_models:
        # Initialize provider dictionary if it doesn't exist
        if provider not in all_results:
            all_results[provider] = {}

        print(f"Testing provider: {provider}, model: {model}")
        try:
            results = test_model(
                documents,
                provider,
                model,
                system_prompt=args.system_prompt,
                user_message=args.user_message,
                batch_size=args.batch_size,
                thinking=args.thinking,
                use_needle=args.use_needle,
                document_type=args.document_type,
                delimiter=args.delimiter,
                reasoning_effort=args.reasoning_effort,
                icl_examples=icl_content,
            )

            # Store results by model name under the provider
            all_results[provider][model] = results

            print(
                f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score"
            )
        except Exception as e:
            print(f"Error testing {provider}/{model}: {str(e)}")
            all_results[provider][model] = {"error": str(e)}

    # Save results
    if not args.output:
        try:
            document_type_clean = (
                "_".join(args.document_type.split())
                if args.document_type
                else "document"
            )
        except:
            pass
        args.output = (
            f"results/{document_type_clean}.json"
            if args.document_type
            else "results/document.json"
        )
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.output}")
    print("Summary:")
    for provider in all_results:
        if provider == "test_date":
            continue
        for model, results in all_results[provider].items():
            if "average_accuracy" in results:
                print(
                    f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score"
                )
            else:
                print(
                    f"{provider} ({model}): Error - {results.get('error', 'unknown error')}"
                )

if __name__ == "__main__":
    main()
