import json
from typing import Dict, Any, List


def write_data(jsonl_file: str, task: Dict[str, Any]):
    with open(jsonl_file, 'a') as f:
        f.write(json.dumps(task) + '\n')


def load_needles(file_name: str) -> List[str]:
    """Load the file and return a list of needles"""
    needles = []
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            # Read each line, strip whitespace, and skip empty lines
            needles = [line.strip() for line in f if line.strip()]
        return needles # explicit return
    except FileNotFoundError:
        print(f"❌ Error: File {file_name} not found")
        return []
    except Exception as e:
        print(f"❌ Error loading needles: {e}")
        return []