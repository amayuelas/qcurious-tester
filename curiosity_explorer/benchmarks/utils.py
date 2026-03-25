"""Shared benchmark utilities: JSONL loading, dataset download, pass@k estimation."""

import json
import math
import os
import urllib.request
from pathlib import Path


def stream_jsonl(path: str):
    """Yield dicts from a JSONL file, one per line.

    Adapted from HumanEval's stream_jsonl pattern.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_if_needed(url: str, dest_path: str) -> str:
    """Download a file if it doesn't already exist. Returns the dest_path."""
    dest = Path(dest_path)
    if dest.exists():
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest} ...")

    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        # Clean up partial downloads
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    print(f"  Downloaded ({dest.stat().st_size / 1024:.1f} KB)")
    return str(dest)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        Estimated pass@k probability.

    Adapted from CRUXEval/HumanEval evaluation code.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
