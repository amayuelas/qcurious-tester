"""Head-to-head on TestGenEval Django examples.

Runs strategies on real Django code inside Docker containers.
The LLM generates test SCRIPTS (not function calls) that get executed
in a full Django environment with coverage tracking.

Usage:
    python run_testgeneval.py --budget 10 --K 3
"""

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import math

import config
from curiosity_explorer.llm import (
    generate_with_model, batch_generate, get_cost, reset_cost,
)
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DOCKER_IMAGE = "aorwall/swe-bench-django_django-testbed:4.0"
SETUP_CODE = "import django; django.setup()"
ENV = {"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"}
WORKING_DIR = "/opt/django__django"

STRATEGIES = ["random", "greedy", "diverse_random", "curiosity_sampling", "curiosity_qvalue"]


def parse_args():
    parser = argparse.ArgumentParser(description="TestGenEval head-to-head")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--S", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=3,
                        help="Number of Django examples to test")
    return parser.parse_args()


def load_django_examples(n=3):
    """Load Django 4.0 examples from TestGenEval-Lite."""
    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]

    django40 = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]

    # Pick examples with moderate baseline coverage (room to improve)
    django40.sort(key=lambda ex: ex["baseline_covs"]["first"])

    examples = []
    for ex in django40[:n]:
        # Determine the module path from code_file
        # django/forms/boundfield.py -> django.forms.boundfield
        module = ex["code_file"].replace("/", ".").replace(".py", "")

        examples.append({
            "instance_id": ex["instance_id"],
            "code_file": ex["code_file"],
            "module": module,
            "code_src": ex["code_src"],
            "test_src": ex["test_src"],
            "baseline_cov": ex["baseline_covs"]["first"],
            "local_imports": ex["local_imports"],
        })

    return examples


# ---------------------------------------------------------------------------
# Test script generation (LLM generates Python scripts, not function calls)
# ---------------------------------------------------------------------------

def _build_context(example, test_history):
    """Build prompt context for test script generation."""
    code = example["code_src"]
    module = example["module"]
    imports = example["local_imports"][:3]

    ctx = f"""You are writing test scripts for this Django module:
File: {example['code_file']}

```python
{code[:3000]}
{"..." if len(code) > 3000 else ""}
```

The module is: {module}
Example imports from test file: {imports[0] if imports else ""}
"""

    if test_history:
        ctx += "\nPrevious test scripts and their results:\n"
        for script, result in test_history[-5:]:
            # Show abbreviated script
            short = script.strip().split("\n")
            if len(short) > 5:
                short = short[:3] + ["..."] + short[-2:]
            ctx += f"  Script:\n"
            for line in short:
                ctx += f"    {line}\n"
            output = result.output[:100] if result.output else ""
            exc = result.exception[:100] if result.exception else ""
            ctx += f"  → output: {output}\n"
            if exc:
                ctx += f"  → exception: {exc}\n"
            ctx += f"  → new_branches: {result.new_branches}\n\n"

    return ctx


SCRIPT_STRATEGIES = [
    "Test EDGE CASES — empty inputs, None values, empty strings, boundary conditions.",
    "Test ERROR HANDLING — pass invalid arguments, wrong types, trigger exceptions.",
    "Test the MAIN FUNCTIONALITY — normal usage with typical, well-formed inputs.",
    "Test a DIFFERENT CLASS or FUNCTION than previous tests focused on. Look at the module and find something untested.",
    "Take the most recent test and MODIFY it slightly — change one argument, use a different method, add a parameter.",
    "Look at the import list and test INTERACTIONS between classes — create instances and pass them to each other.",
    "Test with COMPLEX inputs — nested objects, large data, multiple items, edge-case combinations.",
]


def generate_test_scripts(example, test_history, K=3, diverse=False):
    """Generate K test scripts for a Django module.

    Args:
        diverse: if True, use different prompting strategies per script
    """
    ctx = _build_context(example, test_history)

    base_instructions = """The script runs in a Django environment (django.setup() already called).
Import what you need and exercise the code — print results to verify.

IMPORTANT:
- Do NOT use unittest or pytest — just write executable Python code
- Do NOT import django or call django.setup() — already done
- Focus on behavior NOT covered by previous tests
- Respond with ONLY the Python code, no explanations"""

    if diverse:
        import random as _rng
        strategies = list(SCRIPT_STRATEGIES)
        _rng.shuffle(strategies)
        prompts = []
        for i in range(K):
            strat = strategies[i % len(strategies)]
            prompt = f"""{ctx}

Write a short Python test script (5-15 lines).
{base_instructions}

Strategy: {strat}

```python
"""
            prompts.append(prompt)
    else:
        prompt = f"""{ctx}

Write a short Python test script (5-15 lines) that tests UNTESTED behavior of this module.
{base_instructions}

```python
"""
        prompts = [prompt] * K

    responses = batch_generate(prompts, temperature=0.9, max_tokens=500)

    scripts = []
    for resp in responses:
        script = _parse_script(resp)
        if script and script not in scripts:
            scripts.append(script)

    return scripts


def _parse_script(response):
    """Extract Python code from LLM response."""
    if not response:
        return None

    # Remove markdown fences
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    # Basic validation
    if not code or len(code) < 10:
        return None
    if "import os" in code and "system" in code:
        return None  # safety
    if "subprocess" in code:
        return None

    # Remove any django.setup() that the LLM might add despite instructions
    lines = code.split("\n")
    lines = [l for l in lines if "django.setup()" not in l
             and l.strip() != "import django"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_random(scripts, example, test_history, runner, S):
    return random.choice(scripts)


def select_greedy(scripts, example, test_history, runner, S):
    """Ask LLM which script will cover the most new code."""
    ctx = _build_context(example, test_history)
    script_list = ""
    for i, s in enumerate(scripts):
        short = s.strip()[:200]
        script_list += f"\n  Script {i+1}:\n    {short}\n"

    prompt = f"""{ctx}

Which of these test scripts is MOST LIKELY to cover NEW, untested code paths?

{script_list}

Respond with ONLY the number (1-{len(scripts)})."""

    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def select_curiosity(scripts, example, test_history, runner, S):
    """Select by output prediction entropy."""
    ctx = _build_context(example, test_history)

    scores = {}
    for script in scripts:
        short = script.strip()[:300]
        prompt = f"""{ctx}

What will be the output of this test script?

```python
{short}
```

Respond with ONLY the expected output, nothing else."""

        predictions = batch_generate([prompt] * S, temperature=0.9,
                                     max_tokens=200)
        predictions = [p.strip().lower()[:150] for p in predictions if p]

        if not predictions:
            scores[id(script)] = 0
            continue

        counts = Counter(predictions)
        total = len(predictions)
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
        scores[id(script)] = entropy

    return max(scripts, key=lambda s: scores.get(id(s), 0))


def select_qvalue(scripts, example, test_history, runner, S, gamma=0.5):
    """Select by Q-value with 1-step lookahead."""
    ctx = _build_context(example, test_history)

    q_scores = {}
    for script in scripts:
        short = script.strip()[:300]

        # Immediate info gain
        prompt = f"""{ctx}

What will be the output of this test script?

```python
{short}
```

Respond with ONLY the expected output, nothing else."""

        predictions = batch_generate([prompt] * S, temperature=0.9,
                                     max_tokens=200)
        predictions = [p.strip().lower()[:150] for p in predictions if p]

        if not predictions:
            ig = 0.0
            predicted_output = "unknown"
        else:
            counts = Counter(predictions)
            total = len(predictions)
            ig = -sum((c/total) * math.log2(c/total) for c in counts.values())
            predicted_output = counts.most_common(1)[0][0]

        # Future value: simulate running this script, generate future scripts
        sim_history = list(test_history) + [(script, type('R', (), {
            'output': predicted_output, 'exception': None,
            'new_branches': 0, 'cumulative': runner.get_cumulative_coverage()
        })())]

        future_scripts = generate_test_scripts(example, sim_history, K=2)
        future_value = 0.0

        if future_scripts:
            for fs in future_scripts:
                fs_short = fs.strip()[:300]
                fs_prompt = f"""{ctx}

After running previous tests, what will this new script output?

```python
{fs_short}
```

Respond with ONLY the expected output."""

                fs_preds = batch_generate([fs_prompt] * max(3, S // 2),
                                          temperature=0.9, max_tokens=200)
                fs_preds = [p.strip().lower()[:150] for p in fs_preds if p]

                if fs_preds:
                    fc = Counter(fs_preds)
                    ft = len(fs_preds)
                    fs_ig = -sum((c/ft) * math.log2(c/ft) for c in fc.values())
                    future_value = max(future_value, fs_ig)

        q = ig + gamma * future_value
        q_scores[id(script)] = q

    return max(scripts, key=lambda s: q_scores.get(id(s), 0))


# ---------------------------------------------------------------------------
# Run one strategy on one example
# ---------------------------------------------------------------------------

def run_strategy(example, strategy, budget, K, S, seed):
    """Run a strategy on a Django example, return coverage curve."""
    random.seed(seed)

    runner = DockerCoverageRunner(
        image=DOCKER_IMAGE,
        source_module=example["module"],
        setup_code=SETUP_CODE,
        working_dir=WORKING_DIR,
        env=ENV,
    )

    test_history = []
    curve = []

    for step in range(budget):
        # Use diverse generation for curiosity strategies
        use_diverse = strategy in ("curiosity_sampling", "curiosity_qvalue",
                                   "diverse_random")
        scripts = generate_test_scripts(example, test_history, K=K,
                                        diverse=use_diverse)

        if not scripts:
            curve.append(runner.get_cumulative_coverage())
            continue

        if strategy == "random":
            selected = select_random(scripts, example, test_history, runner, S)
        elif strategy == "diverse_random":
            selected = select_random(scripts, example, test_history, runner, S)
        elif strategy == "greedy":
            selected = select_greedy(scripts, example, test_history, runner, S)
        elif strategy == "curiosity_sampling":
            selected = select_curiosity(scripts, example, test_history, runner, S)
        elif strategy == "curiosity_qvalue":
            selected = select_qvalue(scripts, example, test_history, runner, S)
        else:
            selected = scripts[0]

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

        print(f"      Step {step+1}: new={result.new_branches}, "
              f"cum={runner.get_cumulative_coverage()}", flush=True)

    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("TestGenEval — Django Head-to-Head", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)
    print(f"  Examples: {args.n_examples} Django 4.0 files", flush=True)
    print("=" * 70, flush=True)

    examples = load_django_examples(args.n_examples)
    for ex in examples:
        print(f"  {ex['code_file']} (baseline={ex['baseline_cov']:.0f}%)",
              flush=True)

    test = generate_with_model(config.MODEL, "Say 'ok'", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex['code_file']}", flush=True)

        ex_results = {
            "code_file": ex["code_file"],
            "module": ex["module"],
            "baseline_cov": ex["baseline_cov"],
            "strategies": {},
        }

        # Run strategies sequentially (Docker can't parallelize well)
        for strategy in STRATEGIES:
            print(f"  {strategy}:", flush=True)
            curve = run_strategy(ex, strategy, args.budget, args.K,
                                 args.S, args.seed)
            ex_results["strategies"][strategy] = curve
            final = curve[-1] if curve else 0
            print(f"    final={final}", flush=True)

        all_results.append(ex_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Analysis
    def mean(v):
        return sum(v) / len(v) if v else 0

    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n  {'Strategy':<25} {'Mean Final':>10} {'Mean @5':>10}", flush=True)
    print(f"  {'─' * 47}", flush=True)
    for s in STRATEGIES:
        finals = [r["strategies"][s][-1] for r in all_results if r["strategies"][s]]
        at5 = [r["strategies"][s][4] for r in all_results
               if r["strategies"][s] and len(r["strategies"][s]) >= 5]
        print(f"  {s:<25} {mean(finals):>10.1f} {mean(at5):>10.1f}", flush=True)

    # Per-file
    print(f"\n  Per-file final coverage:", flush=True)
    print(f"  {'File':<40}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s[:10]:>11}", end="")
    print(flush=True)
    for r in all_results:
        print(f"  {r['code_file']:<40}", end="")
        for s in STRATEGIES:
            v = r["strategies"][s][-1] if r["strategies"][s] else 0
            print(f" {v:>11}", end="")
        print(flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "testgeneval_results.json", "w") as f:
        json.dump({
            "config": {"model": config.MODEL, "budget": args.budget,
                       "K": args.K, "S": args.S},
            "results": all_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to results/testgeneval_results.json", flush=True)


if __name__ == "__main__":
    main()
