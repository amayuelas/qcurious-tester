"""Cryptographic corridor functions for benchmarking curiosity-guided exploration.

Gates use RUNTIME LOOKUPS into a seeded permutation table. The LLM can see
the table in the source code but cannot easily determine which inputs pass
without trial and error (the table has 256 entries).

Key properties:
- Gates use T[f(input)] < threshold — must try inputs to discover valid ones
- Error codes are monotonic integers (0, 1, 2...) — LLM learns "I'm getting deeper"
- Each gate tests a different input property via a different table lookup
- Deep logic uses table-derived thresholds — unpredictable branches
"""

import random as _random


def _make_table(seed, size=256):
    """Create a seeded permutation table."""
    rng = _random.Random(seed)
    table = list(range(size))
    rng.shuffle(table)
    return table


def generate_corridor_function(seed, n_gates=4, n_deep_branches=15,
                                gate_difficulty=0.3):
    """Generate a corridor function with runtime table-lookup gates.

    Args:
        seed: Deterministic seed for reproducibility
        n_gates: Number of sequential validation gates (2-8)
        n_deep_branches: Number of branches in deep logic (10-40)
        gate_difficulty: Fraction of inputs that pass each gate (0.1-0.5).
            Higher = easier gates (more inputs pass).

    Returns:
        dict with func_name, source, metadata
    """
    rng = _random.Random(seed)
    func_name = f"func_{seed}"
    table = _make_table(seed)

    lines = []
    lines.append(f"def {func_name}(a, b, c):")
    lines.append(f"    T = {table}")
    lines.append("")

    gate_specs = []
    error_code = 0

    # Gate 0: Type check (always present, readable)
    lines.append("    # Type check")
    lines.append("    if not isinstance(a, str) or not isinstance(b, list) or not isinstance(c, int):")
    lines.append(f"        return {error_code}")
    gate_specs.append({"type": "type_check", "error_code": error_code})
    error_code += 1

    # Gate 1: String length via table lookup
    # T[0] determines the valid length range — LLM must try different lengths
    lines.append("")
    lines.append("    # Gate: string length")
    valid_len_count = max(2, int(10 * gate_difficulty))
    lines.append(f"    if T[len(a) % 256] >= {int(256 * gate_difficulty)}:")
    lines.append(f"        return {error_code}")
    gate_specs.append({"type": "string_length_lookup", "error_code": error_code,
                       "pass_rate": gate_difficulty})
    error_code += 1

    # Build pool of runtime-lookup gates
    gate_pool = []

    # Gate: first char of string via table lookup
    gate_pool.append({
        "code": [
            "    # Gate: first character",
            f"    if T[ord(a[0]) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "first_char_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: integer c via table lookup
    gate_pool.append({
        "code": [
            "    # Gate: integer value",
            f"    if T[(c * 7 + 13) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "int_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: list length via table lookup
    gate_pool.append({
        "code": [
            "    # Gate: list length",
            f"    if T[(len(b) * 11 + 3) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "list_length_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: sum of list elements via table lookup
    gate_pool.append({
        "code": [
            "    # Gate: list sum",
            "    if not b or not all(isinstance(x, int) for x in b):",
            f"        return {error_code}",
            f"    if T[sum(b) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "list_sum_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: ASCII sum of string via table lookup
    gate_pool.append({
        "code": [
            "    # Gate: string content",
            f"    if T[sum(ord(ch) for ch in a) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "ascii_sum_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: combined — uses multiple inputs
    gate_pool.append({
        "code": [
            "    # Gate: combined check",
            f"    if T[(len(a) * len(b) + c) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "combined_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Gate: second char lookup
    gate_pool.append({
        "code": [
            "    # Gate: second character",
            "    if len(a) < 2:",
            f"        return {error_code}",
            f"    if T[(ord(a[1]) * 3) % 256] >= {int(256 * gate_difficulty)}:",
            f"        return {error_code}",
        ],
        "spec": {"type": "second_char_lookup", "error_code": error_code,
                 "pass_rate": gate_difficulty},
    })
    error_code += 1

    # Select gates from pool
    rng.shuffle(gate_pool)
    selected_gates = gate_pool[:max(0, n_gates - 2)]  # -2 because type + length already added

    for gate in selected_gates:
        lines.append("")
        lines.extend(gate["code"])
        gate_specs.append(gate["spec"])

    # Deep logic — uses table-derived thresholds
    lines.append("")
    lines.append("    # Deep logic (passed all gates)")
    lines.append("    v = sum(x * (i + 1) for i, x in enumerate(b)) + c")
    lines.append("    w = sum(ord(ch) for ch in a) + c * len(b)")

    deep_result_start = 100
    result_code = deep_result_start
    remaining = n_deep_branches

    # Primary branches on v with table-derived thresholds
    v_thresholds = sorted(
        [table[30 + i] * 8 + 100 for i in range(min(6, remaining // 2))],
        reverse=True
    )

    for i, tv in enumerate(v_thresholds):
        prefix = "if" if i == 0 else "elif"
        lines.append(f"    {prefix} v > {tv}:")

        # Nested branches on w
        if remaining > 2:
            w_thresh = table[50 + i] * 5 + 50
            lines.append(f"        if w > {w_thresh}:")
            lines.append(f"            if T[(v + w) % 256] < 128:")
            lines.append(f"                return {result_code}")
            result_code += 1
            remaining -= 1
            lines.append(f"            else:")
            lines.append(f"                return {result_code}")
            result_code += 1
            remaining -= 1
            lines.append(f"        else:")
            lines.append(f"            return {result_code}")
            result_code += 1
            remaining -= 1
        else:
            lines.append(f"        return {result_code}")
            result_code += 1
            remaining -= 1

        if remaining <= 1:
            break

    # Additional branches on string/list properties
    if remaining > 2:
        lines.append(f"    elif T[(len(a) + len(b)) % 256] < 128:")
        lines.append(f"        if c % 2 == 0:")
        lines.append(f"            return {result_code}")
        result_code += 1
        remaining -= 1
        lines.append(f"        else:")
        lines.append(f"            return {result_code}")
        result_code += 1
        remaining -= 1

    if remaining > 1:
        lines.append(f"    elif v > 0:")
        lines.append(f"        return {result_code}")
        result_code += 1
        remaining -= 1

    lines.append(f"    else:")
    lines.append(f"        return {result_code}")
    result_code += 1

    source = "\n".join(lines) + "\n"

    n_actual_gates = 2 + len(selected_gates)  # type + length + selected
    total_branches = n_actual_gates * 2 + (result_code - deep_result_start)

    return {
        "func_name": func_name,
        "source": source,
        "metadata": {
            "seed": seed,
            "n_gates": n_actual_gates,
            "n_deep_branches": result_code - deep_result_start,
            "gate_difficulty": gate_difficulty,
            "total_expected_branches": total_branches,
            "gate_specs": gate_specs,
        },
        "difficulty": "unknown",
    }


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = {
    "easy": {"n_gates": 3, "n_deep_branches": 12, "gate_difficulty": 0.5},
    "medium": {"n_gates": 5, "n_deep_branches": 20, "gate_difficulty": 0.3},
    "hard": {"n_gates": 7, "n_deep_branches": 30, "gate_difficulty": 0.15},
}


def generate_benchmark_suite(n_per_level=10, base_seed=42):
    """Generate a full benchmark suite with Easy/Medium/Hard functions."""
    suite = {}
    for level_name, params in DIFFICULTY_LEVELS.items():
        for i in range(n_per_level):
            seed = base_seed * 1000 + hash(level_name) % 100 * 100 + i
            func = generate_corridor_function(seed=seed, **params)
            func["difficulty"] = level_name
            key = f"{level_name}_{i:02d}"
            suite[key] = func
    return suite


def load_cryptographic_programs(n_per_level=10, base_seed=42):
    """Load for use with benchmark runner."""
    suite = generate_benchmark_suite(n_per_level, base_seed)
    programs = {}
    for key, func in suite.items():
        programs[key] = {
            "func_name": func["func_name"],
            "source": func["source"],
            "metadata": {
                "cyclomatic_complexity": func["metadata"]["total_expected_branches"],
                "corridor_depth": func["metadata"]["n_gates"],
                "difficulty": func["difficulty"],
                **func["metadata"],
            },
        }
    return programs
