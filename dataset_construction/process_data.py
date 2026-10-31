'''
This file contains the code to generate datasets for absence-bench given a user created jsonl.
The user should specify an input jsonl file with the following specifications:
    - Each line represents a single "document"
    - Contains a `lines` field that stores a list
    - Optionally, a user may have an 'id' field and a 'metadata' field, which will persist into the new data
    - Every other field is unused and will not be present in the new data


Following the setup of the rest of the repo, datasets will be generated as jsonl files with the following fields:
'id'
'original_context': original_document (str)
'modified_context': modified_document (str)
'omitted_context': omitted_lines (list)
'omitted_index': omitted_indices (list)
'metadata': document_metadata (dict)
'''

import json
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import write_data

# def process_file(input_file: str, omission_prob: float = 0.1,
#                  delimiter: str = '\n', use_needle: bool = False,
#                  needles_file: str = None,
#                  max_lines = None, use_placeholders: bool = False,
#                  document_type: str = None, output_dir: str = None):
#     '''
#     Process a user specified JSONL file to create a dataset with each
#     document having both original and modified versions in the same
#     JSON object
#     '''

# # If testing NIAH setting, load needles from file
#     if use_needle:
#         print(f"Testing the NIAH setting under {document_type} domain")
#         # Use needles_file passed into the function
#         if not needles_file:
#             raise ValueError("Using --use_needle requires providing a valid --needles_file path.")
            
#         needles = load_needles(needles_file)
#     else:
#         needles = None

#     if use_placeholders:
#         print(f"Testing placeholder. Default: <missing lines>")
#         placeholder = "<missing line>"
#     else:
#         placeholder = None

#     # Determine output file path at the start to enable cleanup
#     if output_dir:
#         output_file_path = f'{output_dir}/processed_data.jsonl'
#     else:
#         # Build path based on document type
#         if document_type is None:
#             document_type = 'document'
#         try:
#             document_type_clean = '_'.join(document_type.split())
#         except:
#             document_type_clean = 'document'

#         if use_needle:
#             output_file_path = f'data/{document_type_clean}_needles.jsonl'
#         elif use_placeholders:
#             output_file_path = f'data/{document_type_clean}_placeholder.jsonl'
#         else:
#             output_file_path = f'data/{document_type_clean}.jsonl'

#     # Clean up existing output file to prevent duplicate appends from multiple runs
#     if os.path.exists(output_file_path):
#         print(f"Removing existing output file: {output_file_path}")
#         os.remove(output_file_path)

#     print("===== Processing Data =====")

#     with open(input_file, 'r', encoding='utf-8') as f_in:
#         for line_num, line in tqdm(enumerate(f_in)):
#             try:
#                 # Parse JSON object
#                 document_data = json.loads(line)

#                 if 'lines' not in document_data:
#                     print(f'Warning: Line {line_num + 1} - Missing required \'lines\' field. Skipping.')
#                     continue

#                 prelim_document_lines = document_data['lines'] # lines is required

#                 document_lines = [line.strip() for line in prelim_document_lines if line.strip() != '']

#                 if max_lines is not None:
#                     document_lines = document_lines[:max_lines]
                
#                 original_document = delimiter.join(document_lines)
                
#                 # create modified document w/ omissions
#                 kept_lines = []
#                 omitted_lines = []
#                 omitted_indices = []
#                 # Keep track of used needles if any
#                 used_needles = []
#                 needle_depths = []

#                 for i, document_line in enumerate(document_lines):
#                     if random.random() > omission_prob:
#                         kept_lines.append(document_line)
#                     else:
#                         if use_needle:
#                             kept_lines.append(document_line)
#                             remain_needles = [n for n in needles if n not in used_needles]
#                             if remain_needles:
#                                 needle = random.choice(remain_needles)
#                                 kept_lines.append(needle)
#                                 used_needles.append(needle)

#                                 depth = i / len(document_lines)
#                                 needle_depths.append(depth)
#                             else:
#                                 needle = ""
#                         elif placeholder:
#                             kept_lines.append(placeholder)
                        
#                         # TODO: see if this should be in an else statement.
#                         else:
#                             omitted_lines.append(document_line)
#                             omitted_indices.append(i) # store idx of omitted line
                
#                 # Create modified document w/ omissions omitted
#                 modified_document = delimiter.join(kept_lines)

#                 # Create metadata for final combined entry
#                 document_metadata = (
#                     document_data.get('metadata')
#                     or document_data.get('_metadata')
#                     or {}
#                 )


#                 document_metadata['document_length'] = len(document_lines)
#                 document_metadata['omission_probability'] = omission_prob
#                 document_metadata['n_omitted'] = len(omitted_indices)
#                 document_metadata['delimiter'] = delimiter

#                 # Create a combined entry with both versions
#                 combined_entry = {
#                     '_id': document_data['id'] if 'id' in document_data \
#                             else document_data['_id'] if '_id' in document_data \
#                             else line_num,
#                     'original_context': original_document,
#                     '_modified_context': modified_document,
#                     '_omitted_context': omitted_lines,
#                     'omitted_index': omitted_indices,
#                     'needle_depths': needle_depths,
#                     '_metadata': document_metadata
#                 }

#                 # Add experiment-specific fields
#                 if use_needle:
#                     combined_entry['needles'] = used_needles
#                 elif use_placeholders:
#                     combined_entry['placeholder'] = placeholder

#                 # Write the combined entry to the output file
#                 write_data(output_file_path, combined_entry)


#             except json.JSONDecodeError:
#                 print(f"Warning: Line {line_num+1} - Invalid JSON. Skipping.")
#                 continue
#             except Exception as e:
#                 import pdb; pdb.set_trace()
#                 print(f"Error processing line {line_num+1}: {e}")
#                 continue
    
#     print(f"Process complete. Output saved to {output_file_path}")
#     return output_file_path

def process_file(input_file: str, omission_prob: float = 0.1,
                 delimiter: str = '\n', use_needle: bool = False,
                 needles_file: str = None,
                 max_lines = None, use_placeholders: bool = False,
                 document_type: str = None, output_dir: str = None):
    '''
    Probability-driven mode: decides omissions/needles using omission_prob.
    '''

    # 1. Safely load needles from file
    needles = []
    if use_needle:
        if not needles_file:
            raise ValueError("Using --use_needle requires providing a valid --needles_file path.")
        try:
            with open(needles_file, 'r', encoding='utf-8') as f:
                needles = [line.strip() for line in f if line.strip()]
            print(f"✅ Successfully loaded {len(needles)} needles.")
            if not needles:
                raise ValueError(
                    f"Needles file is empty: {needles_file}. "
                    "NIAH mode requires at least one non-empty needle line."
                )
        except Exception as e:
            raise RuntimeError(
                f"❌ Error loading needles from '{needles_file}': {e}\n"
                "Aborting to avoid silently falling back to omission mode."
            ) from e

    placeholder = "<missing line>" if use_placeholders else None

    # 2. Define output path outside loop
    if output_dir:
        output_file_path = f'{output_dir}/processed_data.jsonl'
    else:
        document_type_clean = '_'.join((document_type or 'document').split())
        suffix = "_needles" if use_needle else "_placeholder" if use_placeholders else ""
        output_file_path = f'data/{document_type_clean}{suffix}.jsonl'

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    print("===== Processing Data (Probability Mode) =====")

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in tqdm(enumerate(f_in)):
            try:
                document_data = json.loads(line)
                prelim_document_lines = document_data.get('lines', [])
                document_lines = [l.strip() for l in prelim_document_lines if l.strip() != '']
                if max_lines: document_lines = document_lines[:max_lines]
                
                original_document = delimiter.join(document_lines)
                
                kept_lines = []
                omitted_lines = []
                omitted_indices = []
                used_needles = []
                needle_depths = []

                # 3. Probability-driven omission/needle logic
                for i, document_line in enumerate(document_lines):
                    # Decide whether this line triggers omission behavior
                    if random.random() < omission_prob:
                        if use_needle and needles:
                            # Needle mode: keep original line and insert one needle
                            kept_lines.append(document_line)
                            selected_needle = random.choice(needles)
                            kept_lines.append(selected_needle)
                            
                            # Track inserted needle info
                            used_needles.append(selected_needle)
                            needle_depths.append(i / len(document_lines))
                        elif placeholder:
                            # Placeholder mode
                            kept_lines.append(placeholder)
                        else:
                            # Default absence mode: omit line
                            omitted_lines.append(document_line)
                            omitted_indices.append(i)
                    else:
                        # Keep line unchanged
                        kept_lines.append(document_line)

                modified_document = delimiter.join(kept_lines)

                # 4. Build metadata
                document_metadata = document_data.get('metadata', {})
                document_metadata.update({
                    'document_length': len(document_lines),
                    'omission_probability': omission_prob,
                    'n_needles': len(used_needles) if use_needle else 0,
                    'n_omitted': len(omitted_indices)
                })

                combined_entry = {
                    '_id': document_data.get('id', line_num),
                    'original_context': original_document,
                    '_modified_context': modified_document,
                    '_omitted_context': omitted_lines,
                    'omitted_index': omitted_indices,
                    'needle_depths': needle_depths,
                    'needles': used_needles, # all inserted needles
                    '_metadata': document_metadata
                }
                
                write_data(output_file_path, combined_entry)

            except Exception as e:
                print(f"Error processing line {line_num+1}: {e}")
    
    return output_file_path

def main():
    parser = argparse.ArgumentParser(
        description='Process dataset to create original and modified versions.'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input file')
    parser.add_argument('-p', '--prob', type=float, default=0.1,
                        help='Probability of omitting a line (default: 0.1)')
    parser.add_argument('-d', '--delimiter', type=str, default='\n',
                        help='Delimiter to set for modified documents (default: newline)')
    parser.add_argument('-m', '--max_lines', type=int,
                        help='Maximum number of lines in the poem')
    parser.add_argument('--document_type', type=str, default='document',
                        help='Name of document type (used in output file naming)')
    parser.add_argument('--use_needle', action="store_true",
                        help='experiment with needle in a haystack')
    parser.add_argument('--use_placeholders', action="store_true",
                        help='use placeholders to help identify omissions')
    parser.add_argument('--random_seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed data (default: data/)')
    
    parser.add_argument('--needles_file', type=str, help='Path to the needles file')

    args = parser.parse_args()
    random.seed(args.random_seed)
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist!")
        return
    
    # Process the input file
    return process_file(
        args.input_file,
        omission_prob=args.prob,
        delimiter=args.delimiter,
        max_lines=args.max_lines,
        use_needle=args.use_needle,
        needles_file=args.needles_file,
        use_placeholders=args.use_placeholders,
        document_type=args.document_type,
        output_dir=args.output_dir)


if __name__ == "__main__":
    main()