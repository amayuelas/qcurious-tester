"""Corridor detection and stratified results analysis."""

import ast
import textwrap


def compute_complexity_metrics(source_code: str) -> dict:
    """Compute cyclomatic complexity and max nesting depth."""
    try:
        tree = ast.parse(textwrap.dedent(source_code))
    except SyntaxError:
        return {"cyclomatic_complexity": 0, "max_nesting_depth": 0}

    complexity = 1  # base
    max_depth = 0

    def _walk(node, depth=0):
        nonlocal complexity, max_depth
        max_depth = max(max_depth, depth)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
                _walk(child, depth + 1)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                _walk(child, depth)
            else:
                _walk(child, depth)

    _walk(tree)
    return {"cyclomatic_complexity": complexity, "max_nesting_depth": max_depth}


def classify_corridor_structure(source_code: str) -> dict:
    """Classify a function's corridor structure.

    Returns dict with complexity metrics and structure classification.
    """
    metrics = compute_complexity_metrics(source_code)

    if metrics["max_nesting_depth"] >= 3 and metrics["cyclomatic_complexity"] > 10:
        structure = "deep_corridor"
    elif metrics["max_nesting_depth"] >= 2:
        structure = "moderate_corridor"
    elif metrics["cyclomatic_complexity"] > 5:
        structure = "branchy_flat"
    else:
        structure = "simple"

    return {**metrics, "structure": structure}


def stratify_results(results: dict, programs: dict) -> dict:
    """Stratify experiment results by corridor structure."""
    stratified = {
        "deep_corridor": [],
        "moderate_corridor": [],
        "branchy_flat": [],
        "simple": [],
    }

    for prog_name, prog_info in programs.items():
        structure = classify_corridor_structure(prog_info["source"])
        if prog_name in results:
            entry = {"name": prog_name, **results[prog_name], **structure}
            stratified[structure["structure"]].append(entry)

    return stratified
