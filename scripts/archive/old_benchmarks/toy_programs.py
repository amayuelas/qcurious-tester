"""Toy programs with known corridor structure for Phase 0 regression tests."""

import textwrap

TOY_PROGRAMS = {
    "corridor_basic": {
        "func_name": "process_order",
        "description": "Basic corridor: validation gates → deep business logic",
        "source": textwrap.dedent('''\
            def process_order(order):
                """Process an order dict. Returns a result dict."""
                # Gate 1: type check (corridor)
                if not isinstance(order, dict):
                    return {"error": "not_a_dict"}

                # Gate 2: required fields (corridor)
                if "items" not in order:
                    return {"error": "missing_items"}
                if "customer_id" not in order:
                    return {"error": "missing_customer_id"}

                # Gate 3: type validation (corridor)
                if not isinstance(order["items"], list) or len(order["items"]) == 0:
                    return {"error": "invalid_items"}
                if not isinstance(order["customer_id"], int):
                    return {"error": "invalid_customer_id"}

                # Deep logic (the clique - this is what we want to reach)
                total = 0
                for item in order["items"]:
                    if not isinstance(item, dict):
                        return {"error": "invalid_item"}
                    price = item.get("price", 0)
                    qty = item.get("quantity", 1)
                    total += price * qty

                if total > 1000:
                    if order["customer_id"] > 100:
                        return {"status": "premium_large_order", "total": total}
                    else:
                        return {"status": "large_order", "total": total, "review": True}
                elif total > 100:
                    return {"status": "medium_order", "total": total}
                elif total > 0:
                    return {"status": "small_order", "total": total}
                else:
                    return {"error": "empty_order"}
        '''),
        "expected_branches": 18,
        "corridor_depth": 4,
    },

    "nested_conditions": {
        "func_name": "classify_triangle",
        "description": "Nested conditions with multiple paths",
        "source": textwrap.dedent('''\
            def classify_triangle(a, b, c):
                """Classify a triangle by its sides. Returns a string."""
                # Gate: must be numeric
                if not all(isinstance(x, (int, float)) for x in [a, b, c]):
                    return "invalid_type"

                # Gate: must be positive
                if a <= 0 or b <= 0 or c <= 0:
                    return "invalid_negative"

                # Gate: triangle inequality
                if a + b <= c or a + c <= b or b + c <= a:
                    return "not_a_triangle"

                # Deep logic
                sides = sorted([a, b, c])
                if sides[0] == sides[2]:
                    return "equilateral"
                elif sides[0] == sides[1] or sides[1] == sides[2]:
                    # Right triangle check
                    if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-9:
                        return "isosceles_right"
                    return "isosceles"
                else:
                    if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-9:
                        return "scalene_right"
                    elif sides[0]**2 + sides[1]**2 < sides[2]**2:
                        return "scalene_obtuse"
                    else:
                        return "scalene_acute"
        '''),
        "expected_branches": 14,
        "corridor_depth": 3,
    },

    "flat_no_corridor": {
        "func_name": "fizzbuzz_extended",
        "description": "Flat branching, no corridor - control condition",
        "source": textwrap.dedent('''\
            def fizzbuzz_extended(n):
                """Extended fizzbuzz. Returns a string."""
                if not isinstance(n, int):
                    return "error"
                if n % 15 == 0:
                    return "fizzbuzz"
                elif n % 3 == 0:
                    return "fizz"
                elif n % 5 == 0:
                    return "buzz"
                elif n % 7 == 0:
                    return "bazz"
                elif n < 0:
                    return "negative"
                elif n == 0:
                    return "zero"
                else:
                    return str(n)
        '''),
        "expected_branches": 8,
        "corridor_depth": 0,
    },

    "computed_branches": {
        "func_name": "analyze_sequence",
        "description": "Computed branches: behavior depends on mathematical properties of input list",
        "source": textwrap.dedent('''\
            def analyze_sequence(seq):
                """Analyze a numeric sequence. Returns classification dict."""
                if not isinstance(seq, list):
                    return {"error": "not_a_list"}
                if len(seq) == 0:
                    return {"error": "empty"}
                if not all(isinstance(x, (int, float)) for x in seq):
                    return {"error": "non_numeric"}
                if len(seq) > 20:
                    return {"error": "too_long"}

                n = len(seq)
                total = sum(seq)
                mean = total / n

                # Property 1: monotonicity
                increasing = all(seq[i] <= seq[i+1] for i in range(n-1))
                decreasing = all(seq[i] >= seq[i+1] for i in range(n-1))

                # Property 2: is it arithmetic?
                if n >= 2:
                    diff = seq[1] - seq[0]
                    is_arithmetic = all(seq[i+1] - seq[i] == diff for i in range(n-1))
                else:
                    is_arithmetic = True
                    diff = 0

                # Property 3: is it geometric?
                if n >= 2 and seq[0] != 0:
                    ratio = seq[1] / seq[0]
                    is_geometric = all(
                        seq[i] != 0 and abs(seq[i+1] / seq[i] - ratio) < 1e-9
                        for i in range(n-1)
                    )
                else:
                    is_geometric = False
                    ratio = 0

                # Property 4: variance
                variance = sum((x - mean) ** 2 for x in seq) / n

                # Classification tree based on computed properties
                if is_arithmetic and is_geometric and n >= 2:
                    return {"type": "constant", "value": seq[0]}

                if is_arithmetic:
                    if diff > 0:
                        if diff == 1 and seq[0] == 1:
                            return {"type": "natural_numbers", "up_to": seq[-1]}
                        return {"type": "arithmetic_increasing", "diff": diff, "start": seq[0]}
                    elif diff < 0:
                        return {"type": "arithmetic_decreasing", "diff": diff, "start": seq[0]}
                    else:
                        return {"type": "constant", "value": seq[0]}

                if is_geometric:
                    if ratio > 1:
                        return {"type": "geometric_growing", "ratio": ratio}
                    elif 0 < ratio < 1:
                        return {"type": "geometric_shrinking", "ratio": ratio}
                    elif ratio < 0:
                        return {"type": "geometric_alternating", "ratio": ratio}
                    else:
                        return {"type": "geometric_zero", "ratio": ratio}

                # Non-arithmetic, non-geometric
                if increasing:
                    if variance < 1:
                        return {"type": "nearly_constant_increasing", "variance": variance}
                    return {"type": "irregular_increasing", "variance": variance}
                elif decreasing:
                    if variance < 1:
                        return {"type": "nearly_constant_decreasing", "variance": variance}
                    return {"type": "irregular_decreasing", "variance": variance}
                else:
                    # Oscillating
                    peaks = sum(1 for i in range(1, n-1) if seq[i] > seq[i-1] and seq[i] > seq[i+1])
                    if peaks == 0:
                        return {"type": "noisy_flat", "mean": mean}
                    elif peaks == 1:
                        return {"type": "unimodal", "peak": max(seq)}
                    elif peaks <= n // 3:
                        return {"type": "low_frequency", "peaks": peaks}
                    else:
                        return {"type": "high_frequency", "peaks": peaks}
        '''),
        "expected_branches": 35,
        "corridor_depth": 4,
    },
}
