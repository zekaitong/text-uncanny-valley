import os
import sys
import argparse
import re

def generate_configs(source_path, ori_input=None, with_needle_wa=False):
    # 1. Validate path
    if not os.path.exists(source_path):
        print(f"Error: file not found: '{source_path}'")
        return

    source_abs_path = os.path.abspath(source_path)
    source_dir = os.path.dirname(source_abs_path)
    
    with open(source_abs_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Working directory: {source_dir}")

    # --- Task A: generate 2.yaml to 10.yaml ---
    print("Generating 2.yaml to 10.yaml...")
    for i in range(2, 11):
        # 1. Replace input_file numeric suffix before .jsonl
        new_content = re.sub(r'(input_file:.*?)1(?=\.jsonl)', rf'\g<1>{i}', content)
        
        # 2. Replace trailing /1 in output_dir
        new_content = re.sub(r'(output_dir:.*?)/1(?=["\']?\s*\n|$)', rf'\g<1>/{i}', new_content)
        
        # 3. Needle-specific replacement when enabled
        if with_needle_wa:
            # Replace wa1 -> wa{i} in needles_file path
            new_content = re.sub(r'(needles_file:.*?)wa1(?=\.txt)', rf'\g<1>wa{i}', new_content)
        
        target_path = os.path.join(source_dir, f"{i}.yaml")
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    print("Done generating 2-10.yaml.")

    # --- Task B: generate ori.yaml when ori_input is provided ---
    if ori_input:
        print("Generating ori.yaml...")
        # 1. Replace input_file path
        ori_content = re.sub(r'(input_file:\s*["\']?)([^"\n\']*)(["\']?)', rf'\1{ori_input}\3', content)

        # 2. Replace trailing /1 in output_dir with /ori
        ori_content = re.sub(r'(output_dir:.*?)/1(?=["\']?\s*\n|$)', r'\1/ori', ori_content)
        
        # 3. Needle-specific replacement when enabled
        if with_needle_wa:
            # Remove _wa1 suffix in needles_file path
            ori_content = re.sub(r'(needles_file:.*?)_wa1(?=\.txt)', r'\1', ori_content)
            
        target_path = os.path.join(source_dir, "ori.yaml")
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(ori_content)
        print(f"Done generating ori.yaml -> {ori_input}")

def main():
    parser = argparse.ArgumentParser(description="Batch YAML generator for AbsenceBench-style configs")
    
    # Required argument: path to template 1.yaml
    parser.add_argument("template", help="Path to 1.yaml")
    
    # Optional: original input path for ori.yaml generation
    parser.add_argument("--ori_input", help="Path to original full input. If set, also generates ori.yaml")
    
    # Optional: apply needle wa index replacements
    parser.add_argument("--with_needle_wa", action="store_true", help="Also update wa indices in needles_file paths")
    
    args = parser.parse_args()
    generate_configs(args.template, args.ori_input, args.with_needle_wa)

if __name__ == "__main__":
    main()