"""CuriosityBench: Real-repo benchmark runner.

Tests exploration strategies on real Python library code inside Docker.
Each target is a source file from a popular pip-installed package.

Usage:
    python run_curiositybench.py --n-targets 5 --budget 8    # quick test
    python run_curiositybench.py --n-targets 35 --budget 10  # full run
    python run_curiositybench.py --repo click                # just click
"""

import argparse
import json
import logging
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import (
    generate_with_model, batch_generate, get_cost, reset_cost,
)
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.benchmarks.curiosity_bench.real_repos import (
    get_targets, DOCKER_IMAGE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "curiosity_qvalue", "reflective_qvalue"]

SCRIPT_STRATEGIES = [
    "Test EDGE CASES — empty inputs, None values, boundary conditions.",
    "Test ERROR HANDLING — invalid arguments, wrong types, trigger exceptions.",
    "Test MAIN FUNCTIONALITY — normal usage with typical inputs.",
    "Test a DIFFERENT CLASS or METHOD than previous tests.",
    "MODIFY the most recent test — change one argument or method call.",
    "Test INTERACTIONS — create multiple objects and pass them to each other.",
    "Test COMPLEX scenarios — nested objects, chained method calls, callbacks.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="CuriosityBench real-repo runner")
    parser.add_argument("--n-targets", type=int, default=35)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--S", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=None, help="Filter by repo")
    parser.add_argument("--strategies", nargs="+", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _build_context(target, test_history):
    """Build prompt context — show module name, docstring, public API."""
    ctx = f"""You are testing the Python module: {target['module']}
From package: {target['repo']}
Description: {target['description']}

The module is installed and importable. You can import from it directly.
"""
    if test_history:
        ctx += "\nPrevious test scripts and results:\n"
        for script, result in test_history[-5:]:
            short = script.strip()[:150]
            out = (result.output or result.exception or "None")[:80]
            ctx += f"  Script: {short}\n  → {out}\n\n"

    return ctx


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_scripts(target, test_history, K=3, diverse=False):
    ctx = _build_context(target, test_history)

    base = f"""Write a short test script (5-15 lines) that imports from {target['module']}
and exercises its classes/functions. Print results to verify.

IMPORTANT:
- Import what you need from the module
- Do NOT use unittest/pytest — just executable Python
- Focus on behavior NOT covered by previous tests
- Respond with ONLY Python code, no explanations

```python
"""

    if diverse:
        strats = list(SCRIPT_STRATEGIES)
        random.shuffle(strats)
        prompts = [f"{ctx}\nStrategy: {strats[i % len(strats)]}\n\n{base}"
                   for i in range(K)]
    else:
        prompts = [f"{ctx}\n{base}"] * K

    responses = batch_generate(prompts, temperature=0.9, max_tokens=500)
    scripts = []
    for resp in responses:
        script = _parse_script(resp)
        if script and script not in scripts:
            scripts.append(script)
    return scripts


def generate_scripts_reflective(target, test_history, learnings, K=3):
    ctx = _build_context(target, test_history)

    prompt = f"""{ctx}

{"LEARNINGS: " + learnings if learnings else ""}

Based on what you've learned, write a test script (5-15 lines) that imports
from {target['module']} and discovers NEW behavior.

IMPORTANT: Respond with ONLY Python code.

```python
"""
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    scripts = []
    for resp in responses:
        script = _parse_script(resp)
        if script and script not in scripts:
            scripts.append(script)
    return scripts


def _parse_script(response):
    if not response:
        return None
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    if not code or len(code) < 10:
        return None
    if "subprocess" in code or "os.system" in code:
        return None
    return code


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_greedy(scripts, target, test_history):
    ctx = _build_context(target, test_history)
    sl = "\n".join(f"  {i+1}. {s.strip()[:120]}" for i, s in enumerate(scripts))
    prompt = f"{ctx}\nWhich test covers the most NEW code paths?\n\n{sl}\n\nRespond with ONLY the number."
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def select_curiosity(scripts, target, test_history, S=6):
    ctx = _build_context(target, test_history)
    scores = {}
    for script in scripts:
        prompt = f"{ctx}\nWhat will this script output?\n```python\n{script[:300]}\n```\nRespond with ONLY the expected output."
        preds = batch_generate([prompt] * S, temperature=0.9, max_tokens=200)
        preds = [p.strip().lower()[:100] for p in preds if p]
        if preds:
            counts = Counter(preds)
            total = len(preds)
            entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
            scores[id(script)] = entropy
        else:
            scores[id(script)] = 0
    return max(scripts, key=lambda s: scores.get(id(s), 0))


def predict_output(script, target, test_history):
    ctx = _build_context(target, test_history)
    prompt = f"{ctx}\nWhat will this output?\n```python\n{script[:300]}\n```\nRespond with ONLY the expected output."
    return generate_with_model(config.MODEL, prompt, 0.3, 150)


def reflect(target, test_history, script, predicted, actual, learnings):
    ctx = _build_context(target, test_history)
    prompt = (f"{ctx}\nPredicted: {predicted[:200]}\nActual: {actual[:200]}\n\n"
              f"{'Learnings: ' + learnings if learnings else ''}\n"
              f"In 2 sentences: What did you learn? What to test next?")
    r = generate_with_model(config.MODEL, prompt, 0.3, 200)
    return (learnings + "\n" + r)[-500:]


# ---------------------------------------------------------------------------
# Run strategy
# ---------------------------------------------------------------------------

def run_strategy(target, strategy, budget, K, S, seed):
    random.seed(seed)
    runner = DockerCoverageRunner(
        image=DOCKER_IMAGE,
        source_module=target["module"],
        setup_code="",
        working_dir="/opt",
        env={},
    )
    test_history = []
    curve = []
    learnings = ""

    for step in range(budget):
        if strategy == "reflective_qvalue":
            scripts = generate_scripts_reflective(target, test_history, learnings, K)
        elif strategy == "curiosity_qvalue":
            scripts = generate_scripts(target, test_history, K, diverse=True)
        else:
            scripts = generate_scripts(target, test_history, K, diverse=False)

        if not scripts:
            curve.append(runner.get_cumulative_coverage())
            continue

        if strategy == "random":
            selected = random.choice(scripts)
        elif strategy == "greedy":
            selected = select_greedy(scripts, target, test_history)
        elif strategy in ("curiosity_qvalue", "reflective_qvalue"):
            selected = select_curiosity(scripts, target, test_history, S)
        else:
            selected = scripts[0]

        predicted = None
        if strategy == "reflective_qvalue":
            predicted = predict_output(selected, target, test_history)

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

        if strategy == "reflective_qvalue" and predicted:
            actual = result.output or result.exception or "None"
            learnings = reflect(target, test_history, selected, predicted,
                                actual, learnings)

        print(f"      Step {step+1}: new={result.new_branches}, "
              f"cum={runner.get_cumulative_coverage()}", flush=True)

    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()
    strategies = args.strategies or STRATEGIES

    targets = get_targets(args.n_targets, args.repo)

    print("=" * 70, flush=True)
    print("CuriosityBench — Real Repos", flush=True)
    print("=" * 70, flush=True)
    print(f"  Docker: {DOCKER_IMAGE}", flush=True)
    print(f"  Targets: {len(targets)}", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}", flush=True)
    print("=" * 70, flush=True)

    test = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target['module']} ({target['repo']})",
              flush=True)

        target_results = {"module": target["module"], "repo": target["repo"],
                          "strategies": {}}

        for strategy in strategies:
            print(f"  {strategy}:", flush=True)
            curve = run_strategy(target, strategy, args.budget, args.K,
                                 args.S, args.seed)
            final = curve[-1] if curve else 0
            target_results["strategies"][strategy] = {
                "curve": curve, "final": final,
            }
            print(f"    final={final}", flush=True)

        all_results.append(target_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Analysis
    import statistics

    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Module':<40}", end="", flush=True)
    for s in strategies:
        print(f" {s[:10]:>11}", end="")
    print(flush=True)
    print("-" * (40 + 12 * len(strategies)), flush=True)
    for r in all_results:
        print(f"{r['module']:<40}", end="")
        for s in strategies:
            v = r["strategies"].get(s, {}).get("final", 0)
            print(f" {v:>11}", end="")
        print(flush=True)

    print("-" * (40 + 12 * len(strategies)), flush=True)
    for s in strategies:
        vals = [r["strategies"][s]["final"] for r in all_results
                if s in r["strategies"]]
        m = statistics.mean(vals) if vals else 0
        se = statistics.stdev(vals) / len(vals)**0.5 if len(vals) > 1 else 0
        print(f"  {s:<25} mean={m:.1f} +/-{se:.1f}", flush=True)

    # Paired comparison
    print(f"\nPaired vs random:", flush=True)
    from scipy import stats as sp_stats
    for s in strategies:
        if s == "random":
            continue
        deltas = []
        for r in all_results:
            if "random" in r["strategies"] and s in r["strategies"]:
                d = r["strategies"][s]["final"] - r["strategies"]["random"]["final"]
                deltas.append(d)
        if not deltas:
            continue
        md = statistics.mean(deltas)
        se = statistics.stdev(deltas) / len(deltas)**0.5 if len(deltas) > 1 else 0
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        ties = sum(1 for d in deltas if d == 0)
        if len(deltas) > 1 and statistics.stdev(deltas) > 0:
            t, p = sp_stats.ttest_1samp(deltas, 0)
            sig = "*" if p < 0.05 else ""
        else:
            p = 1.0
            sig = ""
        print(f"  {s:<25} D={md:>+6.1f} +/-{se:.1f} "
              f"W={wins} L={losses} T={ties} p={p:.4f} {sig}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "curiositybench/real_repos.json", "w") as f:
        json.dump({"config": {"budget": args.budget, "K": args.K, "S": args.S,
                               "strategies": strategies},
                    "results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"Saved", flush=True)


if __name__ == "__main__":
    main()
