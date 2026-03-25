"""ULT (UnLeakedTestBench) benchmark loader.

Loads functions from the ULT dataset (3,909 Python functions).
Data source: https://github.com/huangd1999/UnLeakedTestBench
Each record has: func_name, code, prompt, task_id, test_list.

Import handling: ULT functions often rely on standard library and numpy
imports. We prepend the same import header used by the ULT evaluation
harness (from Ray/main.py) so functions can execute standalone.
"""

import re
import textwrap
import random as _random
from pathlib import Path

import json as _json

from ..utils import download_if_needed

ULT_URL = (
    "https://raw.githubusercontent.com/huangd1999/UnLeakedTestBench/"
    "main/datasets/ULT.jsonl"
)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "ult"

# Standard import header from ULT's own evaluation harness (Ray/main.py).
# Prepended to every function so external imports resolve at runtime.
ULT_IMPORT_HEADER = textwrap.dedent("""\
    import os, re, math, random, string, warnings, datetime, traceback
    import collections, itertools, functools, operator, copy, json, sys
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import numpy
    from typing import List, Dict, Any, Optional, Union, Tuple, Set
""")


def _prepend_imports(code: str) -> str:
    """Prepend the standard import header, skipping duplicates already in code."""
    return ULT_IMPORT_HEADER + "\n" + code


def _compute_cyclomatic_complexity(source: str) -> int:
    """Lightweight cyclomatic complexity computation."""
    from curiosity_explorer.analysis.corridor_analysis import compute_complexity_metrics
    metrics = compute_complexity_metrics(source)
    return metrics["cyclomatic_complexity"]


def load_ult_functions(
    max_functions=None,
    min_complexity=None,
    max_complexity=None,
    lite=False,
    shuffle=False,
    seed=42,
) -> dict:
    """Load ULT functions for Phase 1 calibration.

    Args:
        max_functions: Limit number of functions loaded (None = all).
        min_complexity: Filter: minimum cyclomatic complexity.
        max_complexity: Filter: maximum cyclomatic complexity.
        lite: If True, load only the first 500 functions (quick experiments).
        shuffle: Shuffle before slicing.
        seed: RNG seed for reproducibility.

    Returns:
        dict mapping keys like "ult_0000" to program dicts.
    """
    data_path = _DATA_DIR / "ULT.jsonl"
    download_if_needed(ULT_URL, str(data_path))

    # ULT dataset is a JSON array (not JSONL despite the extension)
    with open(data_path, "r", encoding="utf-8") as f:
        records = _json.load(f)

    if lite:
        records = records[:500]

    if shuffle:
        rng = _random.Random(seed)
        rng.shuffle(records)

    programs = {}
    for idx, rec in enumerate(records):
        code = rec.get("code", "")
        func_name = rec.get("func_name", "")

        # Fall back to parsing the function name from source
        if not func_name:
            match = re.search(r"def\s+(\w+)\s*\(", code)
            func_name = match.group(1) if match else f"ult_func_{idx}"

        # Prepend standard imports so the function can execute standalone
        full_source = _prepend_imports(code)

        # Compute complexity for optional filtering
        complexity = None
        if min_complexity is not None or max_complexity is not None:
            complexity = _compute_cyclomatic_complexity(code)
            if min_complexity is not None and complexity < min_complexity:
                continue
            if max_complexity is not None and complexity > max_complexity:
                continue

        key = f"ult_{idx:04d}"

        # Lazily compute complexity for metadata if not already done
        if complexity is None:
            complexity = _compute_cyclomatic_complexity(code)

        programs[key] = {
            "func_name": func_name,
            "source": full_source,
            "description": rec.get("prompt", f"ULT {func_name}"),
            "metadata": {
                "task_id": rec.get("task_id", key),
                "cyclomatic_complexity": complexity,
                "prompt": rec.get("prompt", ""),
                "test_list": rec.get("test_list", []),
            },
        }

        if max_functions is not None and len(programs) >= max_functions:
            break

    return programs
