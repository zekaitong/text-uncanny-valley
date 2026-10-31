import json
import re
import argparse
import os

def segment_text(text):
    """
    Split text into lines by sentence boundary or newline and clean segments.
    """
    if not text:
        return []
    # Split on period-space or newline
    segments = re.split(r'\.\s+|\n', text.strip())
    # Remove empty segments and ensure punctuation endings
    cleaned = []
    for s in segments:
        s = s.strip()
        if s:
            if not s.endswith(('.', '?', '!')):
                s += "."
            cleaned.append(s)
    return cleaned

def process_gsm8k(input_path, output_path, n):
    """
    Read GSM8K data and group n QA pairs into one AbsenceBench document.
    """
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        return

    all_entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_entries.append(json.loads(line))

    total_entries = len(all_entries)
    print(f"Loaded {total_entries} raw QA pairs. Target group size n={n}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        current_document_lines = []
        count = 0

        for i, entry in enumerate(all_entries):
            # 1. Extract and segment question
            q_lines = segment_text(entry.get('question', ''))
            
            # 2. Extract and segment answer (remove final #### marker)
            raw_answer = entry.get('answer', '').split('####')[0].strip()
            a_lines = segment_text(raw_answer)
            
            # 3. Append current QA lines to document buffer
            current_document_lines.extend(q_lines + a_lines)
            count += 1

            # 4. Write one document when count reaches n or file ends
            if count == n or i == total_entries - 1:
                # Build AbsenceBench format
                document = {
                    "lines": current_document_lines,
                    "metadata": {
                        "combined_count": count,
                        "source": "gsm8k"
                    }
                }
                f_out.write(json.dumps(document, ensure_ascii=False) + '\n')
                
                # Reset buffer
                current_document_lines = []
                count = 0

    print(f"Done. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GSM8K into AbsenceBench document format")
    parser.add_argument("--input", type=str, default="data/logic/gsm8k/test_ori.jsonl", help="Path to raw GSM8K JSONL")
    parser.add_argument("--output", type=str, default="data/logic/gsm8k/gsm8k_abs_docs.jsonl", help="Output path")
    parser.add_argument("-n", type=int, default=5, help="Number of QA pairs per document")
    
    args = parser.parse_args()
    process_gsm8k(args.input, args.output, args.n)