"""Fair comparison: equalize execution count across all strategies.

Every strategy gets exactly EXEC_BUDGET test executions.
- random: generate K, pick 1, repeat EXEC_BUDGET times
- greedy: generate K, LLM picks 1, repeat EXEC_BUDGET times
- cov_greedy: coverage-aware gen K, pick 1, repeat EXEC_BUDGET times
- cov_planned: plan 3 scripts, execute all 3, repeat EXEC_BUDGET/3 times
- random_covfeedback: standard gen K, random pick, but with coverage map in prompt

This way cov_planned gets the same number of executions as random.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "cov_greedy", "cov_planned", "random_covfeedback"]
EXEC_BUDGET = 24  # every strategy gets exactly 24 test executions
K = 3
PLAN_LENGTH = 3


def gen_standard(ctx, hist, K):
    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"
    prompt = (f"{ctx}\n{h}\nWrite a test script (5-10 lines). "
              f"Import from the module and print results.\n"
              f"IMPORTANT: Django is already configured. "
              f"Do NOT call settings.configure() or django.setup().\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=400)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_standard_with_covmap(ctx, hist, cov_map, K):
    """Standard generation but with coverage map shown — isolates feedback effect."""
    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"
    cov_summary = cov_map.coverage_summary()
    prompt = (f"{ctx}\n\n{cov_summary}\n{h}\n"
              f"Write a test script (5-10 lines) that covers branches NOT YET discovered. "
              f"Import from the module and print results.\n"
              f"IMPORTANT: Django is already configured. "
              f"Do NOT call settings.configure() or django.setup().\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=400)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_coverage_greedy(code, module, hist, cov_map, K):
    code_section = f"```python\n{code[:2500]}\n```"
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"
    prompt = f"""Module: {module}
{code_section}

{cov_summary}

Previous tests:
{history_str}

Write a test script (5-10 lines) that covers branches NOT YET discovered.
Look at the code and target specific uncovered code paths.
Import from the module and print results.

IMPORTANT: Django is already configured. Do NOT call settings.configure() or django.setup().
Respond with ONLY executable Python code.

```python
"""
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_coverage_planned(code, module, hist, cov_map, plan_length=3):
    code_section = f"```python\n{code[:2500]}\n```"
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"
    prompt = f"""Module: {module}
{code_section}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach the
DEEPEST uncovered code paths. Think about what setup is needed:

Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-10 lines each).
Import from the module and print results.

IMPORTANT: Django is already configured. Do NOT call settings.configure() or django.setup().

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
        return gen_coverage_greedy(code, module, hist, cov_map, K)
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
                lines = script.split("\n")
                lines = [l for l in lines if "django.setup()" not in l
                         and l.strip() != "import django"
                         and "settings.configure" not in l]
                script = "\n".join(lines).strip()
                if script:
                    scripts.append(script)
    return scripts


def select_greedy(scripts, ctx, hist):
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    prompt = (f"{ctx}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def run_strategy(ex, strategy):
    _random.seed(42)
    module = ex["code_file"].replace("/", ".").replace(".py", "")
    code = ex["code_src"]
    version = ex.get("version", "4.0")
    ctx = f"Module: {module}\n```python\n{code[:2000]}\n```"

    runner = DockerCoverageRunner(
        image=f"aorwall/swe-bench-django_django-testbed:{version}",
        source_module=module,
        setup_code="import django; django.setup()",
        working_dir="/opt/django__django",
        env={"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
    )

    hist = []
    cov_map = CoverageMap()
    executions = 0

    while executions < EXEC_BUDGET:
        if strategy == "cov_greedy":
            scripts = gen_coverage_greedy(code, module, hist, cov_map, K)
        elif strategy == "cov_planned":
            scripts = gen_coverage_planned(code, module, hist, cov_map, PLAN_LENGTH)
        elif strategy == "random_covfeedback":
            scripts = gen_standard_with_covmap(ctx, hist, cov_map, K)
        else:
            scripts = gen_standard(ctx, hist, K)

        if not scripts:
            executions += 1  # count failed generation as 1 execution
            continue

        if strategy == "cov_planned":
            # Execute the plan, but respect execution budget
            for plan_script in scripts:
                if executions >= EXEC_BUDGET:
                    break
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                executions += 1
                print(f"        exec {executions}/{EXEC_BUDGET}: "
                      f"+{result.new_branches}, cum={runner.get_cumulative_coverage()}",
                      flush=True)
        else:
            # Select one script
            if strategy == "random" or strategy == "random_covfeedback":
                selected = _random.choice(scripts)
            elif strategy == "greedy":
                selected = select_greedy(scripts, ctx, hist)
            elif strategy == "cov_greedy":
                selected = _random.choice(scripts)
            else:
                selected = scripts[0]

            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            print(f"      exec {executions}/{EXEC_BUDGET}: "
                  f"+{result.new_branches}, cum={runner.get_cumulative_coverage()}",
                  flush=True)

    return runner.get_cumulative_coverage()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=5)
    args = parser.parse_args()

    reset_cost()

    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]
    examples = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]
    examples.sort(key=lambda ex: ex["baseline_covs"]["first"])
    examples = examples[:args.n_files]

    print("=" * 70, flush=True)
    print("FAIR COMPARISON — Equalized Execution Budget", flush=True)
    print(f"Exec budget={EXEC_BUDGET} per strategy, K={K}, Files={len(examples)}", flush=True)
    print(f"Strategies: {STRATEGIES}", flush=True)
    print("=" * 70, flush=True)

    t = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    start = time.time()
    all_results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex['code_file']}", flush=True)
        file_results = {"file": ex["code_file"], "strategies": {}}

        for strategy in STRATEGIES:
            print(f"  {strategy}:", flush=True)
            final = run_strategy(ex, strategy)
            file_results["strategies"][strategy] = final
            print(f"    final={final}", flush=True)

        all_results.append(file_results)

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\n{'=' * 70}", flush=True)
    print(f"FAIR COMPARISON (all strategies: {EXEC_BUDGET} executions each)", flush=True)
    print(f"{'File':<40}", end="", flush=True)
    for s in STRATEGIES:
        label = s[:12]
        print(f" {label:>14}", end="")
    print(flush=True)
    print("-" * 110, flush=True)
    for r in all_results:
        print(f"{r['file']:<40}", end="")
        for s in STRATEGIES:
            print(f" {r['strategies'].get(s, 0):>14}", end="")
        print(flush=True)

    import statistics
    print("-" * 110, flush=True)
    for s in STRATEGIES:
        vals = [r["strategies"].get(s, 0) for r in all_results]
        m = statistics.mean(vals)
        print(f"  {s:<20} mean={m:.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "covmap_fair_comparison.json", "w") as f:
        json.dump({"exec_budget": EXEC_BUDGET, "results": all_results,
                    "cost": cost, "elapsed": round(elapsed, 1)}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
