#!/usr/bin/env python3
"""
Transformation script for introducing weird spacing and tabbing.

Introduces various spacing anomalies including:
- Random extra spaces between words
- Tabs instead of spaces
- Multiple spaces/tabs
- Leading/trailing whitespace variations
- Inconsistent indentation

Usage:
    python spacing_transform.py --input data/raw_data/legal_docs.jsonl --mode random_spaces
    python spacing_transform.py --input data/raw_data/*.jsonl --mode tabs
    python spacing_transform.py --input data/raw_data/*.jsonl --mode mixed --intensity 0.3
    python spacing_transform.py --file_list datasets.txt --mode mixed --intensity 0.5
"""

import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
import glob
import random
import re


def random_spaces_transform(line: str, intensity: float) -> str:
    """Add random extra spaces between words."""
    words = line.split()
    result = []

    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:  # Not the last word
            # Decide how many spaces to add
            if random.random() < intensity:
                num_spaces = random.randint(2, 5)
                result.append(' ' * num_spaces)
            else:
                result.append(' ')

    return ''.join(result)


def tabs_transform(line: str, intensity: float) -> str:
    """Replace spaces with tabs randomly."""
    words = line.split()
    result = []

    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:  # Not the last word
            if random.random() < intensity:
                result.append('\t')
            else:
                result.append(' ')

    return ''.join(result)


def leading_spaces_transform(line: str, intensity: float) -> str:
    """Add random leading spaces/tabs."""
    if random.random() < intensity:
        indent_type = random.choice([' ', '\t'])
        indent_amount = random.randint(0, 8)
        return indent_type * indent_amount + line
    return line


def trailing_spaces_transform(line: str, intensity: float) -> str:
    """Add random trailing spaces."""
    if random.random() < intensity:
        trailing_amount = random.randint(1, 5)
        return line + ' ' * trailing_amount
    return line


def mixed_transform(line: str, intensity: float) -> str:
    """Apply a mix of spacing transformations."""
    # Apply transformations probabilistically
    if random.random() < intensity:
        line = random_spaces_transform(line, intensity * 0.5)

    if random.random() < intensity:
        line = tabs_transform(line, intensity * 0.3)

    if random.random() < intensity * 0.7:
        line = leading_spaces_transform(line, 0.8)

    if random.random() < intensity * 0.5:
        line = trailing_spaces_transform(line, 0.8)

    return line


def extreme_spaces_transform(line: str, intensity: float) -> str:
    """
    Add extreme spacing variations - multiple spaces/tabs between every word.
    Makes text very hard to read.
    """
    words = line.split()
    result = []

    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:
            # Random combination of spaces and tabs
            whitespace_types = [' ', '\t', '  ', '\t\t', ' \t']
            num_whitespaces = random.randint(1, 5)
            separator = ''.join(random.choice(whitespace_types) for _ in range(num_whitespaces))
            result.append(separator)

    return ''.join(result)


def intra_word_transform(line: str, intensity: float, all_non_space: bool = False) -> tuple[str, float]:
    """
    Randomly insert spaces between adjacent characters.
    
    If all_non_space is True, any two adjacent non-whitespace characters are targets.
    Otherwise, only alphanumeric character gaps are targeted.
    """
    if intensity == 0.0:
        return line, 0.0

    result = []
    eligible_gaps = 0
    inserted_spaces = 0

    for i in range(len(line)):
        result.append(line[i])
        
        if i + 1 < len(line):
            curr_char = line[i]
            next_char = line[i+1]
            
            # Determine if the gap is eligible based on the all_non_space flag
            if all_non_space:
                is_eligible = not curr_char.isspace() and not next_char.isspace()
            else:
                is_eligible = curr_char.isalnum() and next_char.isalnum()
                
            if is_eligible:
                eligible_gaps += 1
                if random.random() < intensity:
                    result.append(" ")
                    inserted_spaces += 1
                
    # rate represents the percentage of eligible gaps that were fragmented
    rate = inserted_spaces / eligible_gaps if eligible_gaps > 0 else 0.0
    return "".join(result), rate

import re

def custom_intra_word_transform(line: str, n: int = None, m: float = None, p: float = 1.0) -> tuple[str, float]:
    """
    Insert spaces into intra-word gaps.
    
    Args:
        line: input line
        n: target gap index (1-indexed)
        m: target gap quantile (0.0-1.0)
        p: insertion probability after a target gap is selected
    """
    if n is None and m is None:
        return line, 0.0

    parts = re.split(r'(\s+)', line)
    transformed_parts = []
    
    total_eligible_gaps = 0
    total_inserted_spaces = 0

    for part in parts:
        if not part or part.isspace():
            transformed_parts.append(part)
            continue

        num_gaps = len(part) - 1
        if num_gaps < 1:
            transformed_parts.append(part)
            continue

        total_eligible_gaps += num_gaps
        target_index = -1

        # 1. Determine candidate target position
        if n is not None:
            if 1 <= n <= num_gaps:
                target_index = n
        elif m is not None:
            calc_idx = round(num_gaps * m)
            if 1 <= calc_idx <= num_gaps:
                target_index = calc_idx

        # 2. Apply insertion probability p
        do_insert = (target_index != -1) and (random.random() < p)

        if do_insert:
            total_inserted_spaces += 1
            chars = list(part)
            new_word_list = []
            for i in range(len(chars)):
                new_word_list.append(chars[i])
                if i + 1 == target_index:
                    new_word_list.append(" ")
            transformed_parts.append("".join(new_word_list))
        else:
            transformed_parts.append(part)

    rate = total_inserted_spaces / total_eligible_gaps if total_eligible_gaps > 0 else 0.0
    return "".join(transformed_parts), rate


def transform_document(lines: List[str], mode: str, intensity: float, all_non_space: bool = False, 
                       gap_n: int = None, gap_m: float = None, gap_p: float = 1.0) -> tuple[List[str], float]:
    """Apply transformation and return (transformed lines, global fragmentation rate)."""
    
    if mode == 'custom_intra_word':
        res = [custom_intra_word_transform(line, n=gap_n, m=gap_m, p=gap_p) for line in lines]
        avg_rate = sum(r[1] for r in res) / len(lines) if lines else 0.0
        return [r[0] for r in res], avg_rate

    if mode == 'intra_word':
        res = [intra_word_transform(line, intensity, all_non_space) for line in lines]
        avg_rate = sum(r[1] for r in res) / len(lines) if lines else 0.0
        return [r[0] for r in res], avg_rate

    transform_funcs = {
        'random_spaces': random_spaces_transform,
        'tabs': tabs_transform,
        'leading': leading_spaces_transform,
        'trailing': trailing_spaces_transform,
        'mixed': mixed_transform,
        'extreme': extreme_spaces_transform
    }
    func = transform_funcs.get(mode)
    return [func(line, intensity) for line in lines], 0.0


def process_file(input_path: str, output_path: str, mode: str, intensity: float, 
                 all_non_space: bool = False, gap_n: int = None, gap_m: float = None, gap_p: float = 1.0):
    """Process a single JSONL file."""
    print(f"Processing {input_path} with mode '{mode}'...")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(tqdm(f_in), 1):
            try:
                doc = json.loads(line)
                if 'lines' not in doc:
                    continue

                # Pass gap_p into transform call
                transformed_lines, frag_rate = transform_document(
                    doc['lines'], mode, intensity, all_non_space, gap_n, gap_m, gap_p
                )

                new_doc = {'lines': transformed_lines}
                for key in ['id', '_id', 'metadata', '_metadata']:
                    if key in doc:
                        new_doc[key] = doc[key]

                metadata_key = 'metadata' if 'metadata' in new_doc else '_metadata'
                if metadata_key not in new_doc:
                    new_doc[metadata_key] = {}
                
                new_doc[metadata_key]['transformation'] = f'spacing_{mode}'
                
                if mode in ['intra_word', 'custom_intra_word']:
                    new_doc[metadata_key]['word_fragmentation_rate'] = frag_rate
                    if mode == 'custom_intra_word':
                        new_doc[metadata_key]['gap_n'] = gap_n
                        new_doc[metadata_key]['gap_m'] = gap_m
                        new_doc[metadata_key]['gap_p'] = gap_p
                else:
                    new_doc[metadata_key]['spacing_intensity'] = intensity

                f_out.write(json.dumps(new_doc, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='Apply spacing/tabbing transformations')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Input file path or glob')
    input_group.add_argument('--file_list', type=str, help='File containing list of inputs')
    
    parser.add_argument('--mode', type=str, required=True,
                    choices=['random_spaces', 'tabs', 'leading', 'trailing', 'mixed', 'extreme', 'intra_word', 'custom_intra_word'],
                    help='Transformation mode')
    
    parser.add_argument('--intensity', type=float, default=0.5, help='Intensity for random modes')
    
    # Gap-selection parameters for custom intra-word mode
    parser.add_argument('--gap_n', type=int, default=None, help='Target gap index (1-indexed)')
    parser.add_argument('--gap_m', type=float, default=None, help='Target gap quantile (0.0-1.0)')
    parser.add_argument('--gap_p', type=float, default=1.0, help='Probability of insertion (0.0-1.0, default: 1.0)')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--suffix', type=str, default=None, help='Output suffix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all_non_space', action='store_true', help='Target all non-space gaps')

    args = parser.parse_args()
    random.seed(args.seed)

    if args.file_list:
        with open(args.file_list, 'r') as f:
            input_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        print("Error: No files found")
        return

    suffix = args.suffix if args.suffix else f'_spacing_{args.mode}'

    for input_path in input_files:
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            continue

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / (input_path_obj.stem + suffix + input_path_obj.suffix)
        else:
            output_path = input_path_obj.parent / (input_path_obj.stem + suffix + input_path_obj.suffix)

    # Pass all parameters through
        process_file(input_path, str(output_path), args.mode, args.intensity, 
                     args.all_non_space, args.gap_n, args.gap_m, args.gap_p)

if __name__ == "__main__":
    main()