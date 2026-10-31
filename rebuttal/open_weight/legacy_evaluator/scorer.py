"""Load only the frozen scorer functions without importing API providers.

The historical ``test_llms.py`` snapshot also imports provider adapters and
loads ``.env``. Open-weight evaluation needs neither. This module compiles the
two required functions directly from that frozen source, so their function
bodies stay byte-for-byte traceable while scorer import remains credential-free.
"""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


SNAPSHOT = Path(__file__).with_name("test_llms.py")
FUNCTIONS = {"load_documents", "evaluate_response"}


def _load_functions():
    tree = ast.parse(SNAPSHOT.read_text(encoding="utf-8"), filename=str(SNAPSHOT))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in FUNCTIONS
    ]
    if {node.name for node in nodes} != FUNCTIONS:
        raise RuntimeError("Frozen evaluator snapshot is missing required functions")
    namespace = {
        "Any": Any,
        "Counter": Counter,
        "Dict": Dict,
        "List": List,
        "json": json,
        "random": random,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SNAPSHOT), "exec"), namespace)
    return namespace["load_documents"], namespace["evaluate_response"]


load_documents, evaluate_response = _load_functions()

