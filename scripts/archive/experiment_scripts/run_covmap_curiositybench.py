"""Coverage-map test on CuriosityBench (real repos in Docker).

Tests cov_planned vs baselines on non-Django real-world code.
"""

import argparse
import random as _random
import re
import time
import json
import logging

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, _parse_script,
)
from curiosity_explorer.benchmarks.curiosity_bench.real_repos import (
    get_targets, DOCKER_IMAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "cov_greedy", "cov_planned"]
BUDGET = 8
K = 3


def gen_standard(target, hist, K):
    ctx = f"Module: {target['module']} (from {target['repo']})\n{target['description']}"
    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"
    prompt = (f"{ctx}\n{h}\nWrite a test script (5-15 lines). "
              f"Import from the module and print results.\n"
              f"Respond with ONLY executable Python code.\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_coverage_greedy(target, hist, cov_map, K):
    ctx = f"Module: {target['module']} (from {target['repo']})\n{target['description']}"
    cov_summary = cov_map.coverage_summary()

    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    prompt = f"""{ctx}

{cov_summary}

Previous tests:
{history_str}

Write a test script (5-15 lines) that covers branches NOT YET discovered.
Import from the module and print results.
Respond with ONLY executable Python code.

```python
"""
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_coverage_planned(target, hist, cov_map, plan_length=3):
    ctx = f"Module: {target['module']} (from {target['repo']})\n{target['description']}"
    cov_summary = cov_map.coverage_summary()

    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    prompt = f"""{ctx}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach the
DEEPEST uncovered code paths. Think about what setup is needed:

Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-15 lines each).
Import from the module and print results.

Format your response as:
### TEST 1
```python
[code]
```
### TEST 2
```python
[code]
```
### TEST 3
```python
[code]
```
"""
    response = generate_with_model("gemini-3-flash-preview", prompt,
                                    temperature=0.7, max_tokens=1500)
    scripts = _parse_plan(response)
    if not scripts:
        return gen_coverage_greedy(target, hist, cov_map, K)
    return scripts


def _parse_plan(response):
    if not response:
        return []
    scripts = []
    parts = re.split(r'###\s*TEST\s*\d+', response)
    for part in parts:
        code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
        if code_match:
            script = code_match.group(1).strip()
            if script and len(script) > 10:
                scripts.append(script)
    return scripts


def select_greedy(scripts, target, hist):
    ctx = f"Module: {target['module']} (from {target['repo']})\n{target['description']}"
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    prompt = (f"{ctx}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def run_strategy(target, strategy):
    _random.seed(42)
    runner = DockerCoverageRunner(
        image=DOCKER_IMAGE,
        source_module=target["module"],
        setup_code="",
        working_dir="/opt",
        env={},
    )

    hist = []
    cov_map = CoverageMap()

    for step in range(BUDGET):
        if strategy == "cov_greedy":
            scripts = gen_coverage_greedy(target, hist, cov_map, K)
        elif strategy == "cov_planned":
            scripts = gen_coverage_planned(target, hist, cov_map, plan_length=3)
        else:
            scripts = gen_standard(target, hist, K)

        if not scripts:
            continue

        if strategy == "random":
            selected = _random.choice(scripts)
        elif strategy == "greedy":
            selected = select_greedy(scripts, target, hist)
        elif strategy == "cov_greedy":
            selected = _random.choice(scripts)
        elif strategy == "cov_planned":
            for plan_script in scripts:
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                print(f"        plan step: +{result.new_branches}, "
                      f"cum={runner.get_cumulative_coverage()}", flush=True)
            continue

        result = runner.run_test(selected)
        hist.append((selected, result))
        cov_map.update(selected, set(), result.new_branches)

        print(f"      Step {step+1}: new={result.new_branches}, "
              f"cum={runner.get_cumulative_coverage()}", flush=True)

    return runner.get_cumulative_coverage()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    reset_cost()
    targets = get_targets(args.n_targets, args.repo)

    print("=" * 70, flush=True)
    print("CuriosityBench Coverage Map Test", flush=True)
    print(f"Budget={BUDGET}, K={K}, Targets={len(targets)}", flush=True)
    print(f"Strategies: {STRATEGIES}", flush=True)
    print("=" * 70, flush=True)

    t = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    start = time.time()
    all_results = []

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target['module']} ({target['repo']})",
              flush=True)
        file_results = {"module": target["module"], "repo": target["repo"],
                        "strategies": {}}

        for strategy in STRATEGIES:
            print(f"  {strategy}:", flush=True)
            final = run_strategy(target, strategy)
            file_results["strategies"][strategy] = final
            print(f"    final={final}", flush=True)

        all_results.append(file_results)

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\n{'=' * 70}", flush=True)
    print(f"{'Module':<35}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s:>12}", end="")
    print(flush=True)
    print("-" * 85, flush=True)
    for r in all_results:
        print(f"{r['module']:<35}", end="")
        for s in STRATEGIES:
            print(f" {r['strategies'].get(s, 0):>12}", end="")
        print(flush=True)

    import statistics
    print("-" * 85, flush=True)
    for s in STRATEGIES:
        vals = [r["strategies"].get(s, 0) for r in all_results]
        m = statistics.mean(vals)
        print(f"  {s:<15} mean={m:.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "covmap_curiositybench.json", "w") as f:
        json.dump({"results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
