"""Test learning progress approach vs baselines on Django files."""

import random
import math
import re
import time
import json
import logging
from collections import Counter

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.learning_progress import (
    LearningProgressTracker, _parse_script,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "lp_random", "lp_qvalue"]
BUDGET = 8
K = 3
S = 6


def make_runner(module, version="4.0"):
    return DockerCoverageRunner(
        image=f"aorwall/swe-bench-django_django-testbed:{version}",
        source_module=module,
        setup_code="import django; django.setup()",
        working_dir="/opt/django__django",
        env={"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
    )


def build_ctx(code_file, code_src):
    module = code_file.replace("/", ".").replace(".py", "")
    return f"Module: {module}\n```python\n{code_src[:2000]}\n```"


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


def gen_enriched(ctx, hist, tracker, K):
    enriched = tracker.build_enriched_history(hist)
    prompt = (f"{ctx}\n{enriched}\n"
              f"Write a short test script (5-10 lines) that tests behavior "
              f"you haven't confirmed yet. Import from the module and print results.\n"
              f"IMPORTANT: Django is already configured (django.setup() already called). "
              f"Do NOT call settings.configure() or django.setup().\n"
              f"Respond with ONLY executable Python code, no explanations.\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=400)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def select_greedy(scripts, ctx, hist):
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    h = ""
    if hist:
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:40]
            h += f"  {s.strip()[:60]} -> {out}\n"
    prompt = (f"{ctx}\n{h}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def select_qvalue(scripts, ctx, hist, S):
    scores = {}
    for script in scripts:
        prompt = (f"{ctx}\nWhat will this output?\n"
                  f"```python\n{script[:200]}\n```\nOnly the output.")
        preds = batch_generate([prompt] * S, temperature=0.9, max_tokens=150)
        preds = [p.strip().lower()[:80] for p in preds if p]
        if preds:
            counts = Counter(preds)
            total = len(preds)
            ent = -sum((c/total) * math.log2(c/total) for c in counts.values())
            scores[id(script)] = ent
        else:
            scores[id(script)] = 0
    return max(scripts, key=lambda s: scores.get(id(s), 0))


def run_strategy(ex, strategy):
    random.seed(42)
    module = ex["code_file"].replace("/", ".").replace(".py", "")
    ctx = build_ctx(ex["code_file"], ex["code_src"])
    version = ex.get("version", "4.0")

    runner = make_runner(module, version)
    hist = []
    tracker = LearningProgressTracker() if strategy.startswith("lp_") else None

    for step in range(BUDGET):
        if tracker:
            scripts = gen_enriched(ctx, hist, tracker, K)
        else:
            scripts = gen_standard(ctx, hist, K)

        if not scripts:
            continue

        if strategy in ("random", "lp_random"):
            selected = random.choice(scripts)
        elif strategy == "greedy":
            selected = select_greedy(scripts, ctx, hist)
        elif strategy == "lp_qvalue":
            selected = select_qvalue(scripts, ctx, hist, S)
        else:
            selected = scripts[0]

        pred = None
        if tracker:
            pred = tracker.predict(selected, ctx)

        result = runner.run_test(selected)
        hist.append((selected, result))

        if tracker and pred:
            actual = result.output or result.exception or "None"
            tracker.update(selected, pred, actual, ctx)

    final = runner.get_cumulative_coverage()
    return final


def main():
    reset_cost()

    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]
    examples = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]
    examples.sort(key=lambda ex: ex["baseline_covs"]["first"])
    examples = examples[:5]

    print("=" * 70, flush=True)
    print("4-Way: random vs greedy vs lp_random vs lp_qvalue", flush=True)
    print(f"Budget={BUDGET}, K={K}, Files={len(examples)}", flush=True)
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
            final = run_strategy(ex, strategy)
            file_results["strategies"][strategy] = final
            print(f"  {strategy:<15} final={final}", flush=True)

        all_results.append(file_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"{'File':<40}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s:>12}", end="")
    print(flush=True)
    print("-" * 90, flush=True)
    for r in all_results:
        print(f"{r['file']:<40}", end="")
        for s in STRATEGIES:
            print(f" {r['strategies'].get(s, 0):>12}", end="")
        print(flush=True)

    import statistics
    print("-" * 90, flush=True)
    for s in STRATEGIES:
        vals = [r["strategies"].get(s, 0) for r in all_results]
        m = statistics.mean(vals)
        print(f"  {s:<15} mean={m:.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "lp_test_results.json", "w") as f:
        json.dump({"results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
