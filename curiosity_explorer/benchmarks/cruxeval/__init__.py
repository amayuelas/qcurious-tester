"""CRUXEval benchmark loader.

Loads functions from the CRUXEval dataset (800 short Python functions).
Data source: https://github.com/facebookresearch/cruxeval
Each record has: id, code, input, output.
"""

import re
import random as _random
from pathlib import Path

from ..utils import stream_jsonl, download_if_needed

CRUXEVAL_URL = (
    "https://raw.githubusercontent.com/facebookresearch/cruxeval/"
    "main/data/cruxeval.jsonl"
)

# Resolved at import time relative to project root
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "cruxeval"


def load_cruxeval_functions(
    max_functions=None,
    shuffle=False,
    seed=42,
) -> dict:
    """Load CRUXEval functions for calibration validation.

    Args:
        max_functions: Limit number of functions loaded (None = all).
        shuffle: Shuffle before slicing.
        seed: RNG seed for reproducibility.

    Returns:
        dict mapping keys like "cruxeval_000" to program dicts
        with func_name, source, description, and metadata.
    """
    data_path = _DATA_DIR / "cruxeval.jsonl"
    download_if_needed(CRUXEVAL_URL, str(data_path))

    records = list(stream_jsonl(str(data_path)))

    if shuffle:
        rng = _random.Random(seed)
        rng.shuffle(records)

    if max_functions is not None:
        records = records[:max_functions]

    programs = {}
    for idx, rec in enumerate(records):
        code = rec["code"]
        # Extract the function name from `def <name>(`
        match = re.search(r"def\s+(\w+)\s*\(", code)
        func_name = match.group(1) if match else f"f_{idx}"

        # Use a unique key so each function gets its own tempdir
        key = f"cruxeval_{idx:04d}"

        programs[key] = {
            "func_name": func_name,
            "source": code,
            "description": f"CRUXEval #{rec.get('id', idx)}",
            "metadata": {
                "task_id": rec.get("id", f"cruxeval_{idx}"),
                "input": rec.get("input", ""),
                "output": rec.get("output", ""),
            },
        }

    return programs
