"""Black-box head-to-head: LLM sees only function signature + test history.

No source code, no coverage feedback. The purest test of Bayesian exploration.
The LLM must learn program behavior entirely through interaction.

Strategies:
  - random: Pick randomly from K candidates
  - greedy: Ask LLM which candidate will reveal the most (no coverage info available)
  - curiosity_sampling: Pick highest output prediction entropy
  - diverse_random: Diverse prompts + random selection
  - diverse_art: Diverse prompts + most-distant selection
  - oracle: Execute all K, pick best (cheating upper bound)

Usage:
    python run_blackbox.py --targeted --budget 15 --K 5
    python run_blackbox.py --toy
"""

import argparse
import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import (
    generate_with_model, batch_generate, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.runner.trace_parser import extract_function_signature
from curiosity_explorer.explorer.art_selection import select_most_distant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CORRIDOR_NAMES = {
    "isRegexPattern", "artworkAreaR6", "getIntervalRange",
    "optim_lr_sched_updater", "dig_2d", "parse_change", "z_defaults",
    "get_codec_rank_static", "CheckIfItIsHotspot", "score_v2",
}

HIGH_GAP_NAMES = {
    "isCardMatch", "dig_2d", "get_codec_rank_static", "isRegexPattern",
    "CheckIfItIsHotspot", "getIntervalRange", "score_v2", "statusInterpreter",
}

STRATEGIES = [
    "random", "greedy", "curiosity_sampling",
    "diverse_random", "diverse_art", "oracle",
]

# Diverse prompting strategies (black-box adapted — no reference to source code)
DIVERSE_STRATEGIES = [
    "Generate a test with EDGE CASE inputs — boundary values, empty inputs, zero, None.",
    "Generate a test likely to trigger an ERROR — invalid types, missing fields.",
    "Generate a test with TYPICAL, well-formed inputs the function was designed for.",
    "Generate an ADVERSARIAL test — unusual combinations the developer didn't anticipate.",
    "Take the most recent test and MUTATE it — change one argument slightly.",
    "Based on the test history, identify what input patterns haven't been tried and generate one.",
    "Generate a test that is MAXIMALLY DIFFERENT from all previous tests.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Black-box head-to-head")
    parser.add_argument("--num-functions", type=int, default=30)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--min-complexity", type=int, default=20)
    parser.add_argument("--benchmark", default="ult")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--targeted", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Black-box context building (NO source code, NO coverage)
# ---------------------------------------------------------------------------

def _bb_context(func_name: str, source_code: str, test_history: list) -> str:
    """Build black-box context: signature + test history (outputs only)."""
    sig = extract_function_signature(source_code)
    ctx = f"Function signature:\n```python\n{sig}\n```\n"

    if test_history:
        recent = test_history[-8:]
        ctx += "\nPrevious test results:\n"
        for test_code, result in recent:
            output = result.output or result.exception or "None"
            ctx += f"  {test_code} → {output}\n"

    return ctx


# ---------------------------------------------------------------------------
# Black-box candidate generation
# ---------------------------------------------------------------------------

def _bb_generate(func_name, source, test_history, K=5):
    """Generate K candidates in black-box mode (no source code visible)."""
    ctx = _bb_context(func_name, source, test_history)

    prompt = f"""{ctx}

You are testing this function to discover all its behaviors.
You can only see the signature and previous test results — NOT the source code.
Generate a single test call that would reveal NEW behavior not seen in previous tests.
Respond with ONLY the function call, nothing else. Example:
{func_name}(arg1, arg2)"""

    responses = batch_generate([prompt] * K, temperature=0.9)

    candidates = []
    for resp in responses:
        call = _parse(resp, func_name)
        if call and call not in candidates:
            candidates.append(call)
    return candidates


def _bb_generate_diverse(func_name, source, test_history, K=5):
    """Generate K diverse candidates in black-box mode."""
    ctx = _bb_context(func_name, source, test_history)

    selected = list(DIVERSE_STRATEGIES)
    random.shuffle(selected)
    selected = selected[:K]

    prompts = []
    for strat in selected:
        prompt = f"""{ctx}

You are testing this function to discover all its behaviors.
You can only see the signature and previous test results — NOT the source code.

{strat}

Respond with ONLY the function call, nothing else. Example:
{func_name}(arg1, arg2)"""
        prompts.append(prompt)

    responses = batch_generate(prompts, temperature=0.9)

    candidates = []
    for resp in responses:
        call = _parse(resp, func_name)
        if call and call not in candidates:
            candidates.append(call)
    return candidates


def _parse(response, func_name):
    if not response:
        return None
    call = response.strip().split("\n")[0].strip()
    if call.startswith("```"):
        call = call.strip("`").strip()
    if not call.startswith(func_name + "("):
        return None
    if call.count("(") != call.count(")"):
        return None
    return call


# ---------------------------------------------------------------------------
# Black-box selection strategies
# ---------------------------------------------------------------------------

def _bb_select_greedy(candidates, func_name, source, test_history):
    """In black-box, greedy asks 'which test will reveal the most new behavior?'
    (no coverage info available)."""
    ctx = _bb_context(func_name, source, test_history)
    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""{ctx}

Which of these tests is MOST LIKELY to reveal NEW, previously-unseen behavior?
You cannot see the source code — reason from the signature and test history only.

Candidates:
{cand_list}

Respond with ONLY the number (1-{len(candidates)})."""

    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _bb_score_entropy(candidates, func_name, source, test_history, S=8):
    """Score candidates by output prediction entropy in black-box mode."""
    ctx = _bb_context(func_name, source, test_history)

    scores = {}
    for cand in candidates:
        prompt = f"""{ctx}

What will be the output of: {cand}

You cannot see the source code. Predict based on the function signature and previous test results only.
Respond with ONLY the expected output value, nothing else."""

        predictions = batch_generate([prompt] * S, temperature=0.9, max_tokens=100)
        predictions = [p.strip().lower()[:100] for p in predictions if p]

        if not predictions:
            scores[cand] = 0
            continue

        from collections import Counter
        counts = Counter(predictions)
        total = len(predictions)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        scores[cand] = entropy

    return scores


def _bb_select_oracle(candidates, func_name, source, runner):
    """Oracle still executes all (it's a cheating baseline)."""
    best, best_gain = candidates[0], -1
    for c in candidates:
        tr = CoverageRunner(func_name, source)
        tr.cumulative_branches = set(runner.cumulative_branches)
        r = tr.run_test(c)
        if r.new_branches > best_gain:
            best, best_gain = c, r.new_branches
    return best


# ---------------------------------------------------------------------------
# Run one strategy
# ---------------------------------------------------------------------------

def run_strategy(func_name, source, strategy, budget, K, S, seed):
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []

    use_diverse = strategy.startswith("diverse_")

    for step in range(budget):
        if use_diverse:
            candidates = _bb_generate_diverse(func_name, source, test_history, K)
        else:
            candidates = _bb_generate(func_name, source, test_history, K)

        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            continue

        if strategy in ("random", "diverse_random"):
            selected = random.choice(candidates)
        elif strategy == "greedy":
            selected = _bb_select_greedy(candidates, func_name, source,
                                         test_history)
        elif strategy == "curiosity_sampling":
            scores = _bb_score_entropy(candidates, func_name, source,
                                       test_history, S)
            selected = max(candidates, key=lambda c: scores.get(c, 0))
        elif strategy == "diverse_art":
            selected = select_most_distant(candidates, test_history)
        elif strategy == "oracle":
            selected = _bb_select_oracle(candidates, func_name, source, runner)
        else:
            selected = candidates[0]

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("BLACK-BOX Head-to-Head", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Mode: BLACK-BOX (no source code, no coverage feedback)", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)

    # Load functions
    if args.toy:
        from curiosity_explorer.benchmarks.toy_programs import TOY_PROGRAMS
        programs = {}
        for key, prog in TOY_PROGRAMS.items():
            programs[key] = {
                "func_name": prog["func_name"],
                "source": prog["source"],
                "metadata": {"cyclomatic_complexity": prog.get("expected_branches", 0)},
            }
        print(f"  Toy programs: {len(programs)}", flush=True)
    else:
        from curiosity_explorer.benchmarks import load_benchmark
        all_programs = load_benchmark(
            args.benchmark, min_complexity=args.min_complexity, max_functions=500,
        )
        if args.targeted:
            programs = {k: p for k, p in all_programs.items()
                        if p["func_name"] in HIGH_GAP_NAMES}
            print(f"  Targeted: {len(programs)} high-gap functions", flush=True)
        else:
            corridor = {k: p for k, p in all_programs.items()
                        if p["func_name"] in CORRIDOR_NAMES}
            non_all = {k: p for k, p in all_programs.items()
                       if p["func_name"] not in CORRIDOR_NAMES}
            needed = max(0, args.num_functions - len(corridor))
            rng = random.Random(args.seed)
            nc_keys = list(non_all.keys())
            rng.shuffle(nc_keys)
            non_corr = {k: non_all[k] for k in nc_keys[:needed]}
            programs = {}
            programs.update(corridor)
            programs.update(non_corr)

    n_corr = sum(1 for p in programs.values() if p["func_name"] in CORRIDOR_NAMES)
    print(f"  Functions: {len(programs)} (corridor={n_corr})", flush=True)
    print("=" * 70, flush=True)

    test = generate_with_model(config.MODEL, "Say 'ok'", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        is_corridor = func_name in CORRIDOR_NAMES
        tag = "CORR" if is_corridor else "other"

        print(f"\n[{i+1}/{len(programs)}] {key} ({func_name}, {tag})", flush=True)

        func_results = {
            "func_key": key, "func_name": func_name,
            "is_corridor": is_corridor, "strategies": {},
        }

        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, s,
                                args.budget, args.K, args.S, args.seed): s
                for s in STRATEGIES
            }
            for future in as_completed(futures):
                s = futures[future]
                try:
                    curve = future.result()
                except Exception as e:
                    log.warning(f"{s} failed: {e}")
                    curve = []
                func_results["strategies"][s] = curve
                final = curve[-1] if curve else 0
                print(f"  {s:<25} final={final}", flush=True)

        all_results.append(func_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Analysis
    def mean(v):
        return sum(v) / len(v) if v else 0

    for subset_name, pred in [("OVERALL", lambda r: True),
                               ("CORRIDOR", lambda r: r["is_corridor"]),
                               ("NON-CORRIDOR", lambda r: not r["is_corridor"])]:
        subset = [r for r in all_results if pred(r)]
        if not subset:
            continue
        print(f"\n{'=' * 70}", flush=True)
        print(f"{subset_name} (n={len(subset)})", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  {'Strategy':<25} {'Final':>7} {'@5':>7} {'@10':>7}", flush=True)
        print(f"  {'─' * 50}", flush=True)
        for s in STRATEGIES:
            finals = [r["strategies"][s][-1] for r in subset if r["strategies"][s]]
            at5 = [r["strategies"][s][4] for r in subset
                   if r["strategies"][s] and len(r["strategies"][s]) >= 5]
            at10 = [r["strategies"][s][9] for r in subset
                    if r["strategies"][s] and len(r["strategies"][s]) >= 10]
            print(f"  {s:<25} {mean(finals):>7.1f} {mean(at5):>7.1f} "
                  f"{mean(at10):>7.1f}", flush=True)

    # Win rates
    print(f"\n{'=' * 70}", flush=True)
    print("WIN RATES vs GREEDY (black-box greedy has NO coverage info)", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cmp in STRATEGIES:
        if cmp == "greedy":
            continue
        for sn, pred in [("ALL", lambda r: True), ("CORR", lambda r: r["is_corridor"])]:
            subset = [r for r in all_results if pred(r)]
            if not subset:
                continue
            w = l = t = 0
            for r in subset:
                a = r["strategies"][cmp][-1] if r["strategies"][cmp] else 0
                b = r["strategies"]["greedy"][-1] if r["strategies"]["greedy"] else 0
                if a > b: w += 1
                elif a < b: l += 1
                else: t += 1
            n = w + l + t
            if n:
                print(f"  {cmp:<25} [{sn:<5}] W={w:>2} L={l:>2} T={t:>2} "
                      f"(win={w/n:.0%})", flush=True)

    # Per-function table
    print(f"\n{'=' * 70}", flush=True)
    print("PER-FUNCTION", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'Function':<25} {'Tag':<5}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s[:10]:>11}", end="")
    print(flush=True)
    for r in all_results:
        tag = "CORR" if r["is_corridor"] else ""
        print(f"  {r['func_name']:<25} {tag:<5}", end="")
        for s in STRATEGIES:
            v = r["strategies"][s][-1] if r["strategies"][s] else 0
            print(f" {v:>11}", end="")
        print(flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "toy" if args.toy else ("targeted" if args.targeted else "ult")
    out_path = config.RESULTS_DIR / f"blackbox_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"model": config.MODEL, "mode": "black-box",
                       "budget": args.budget, "K": args.K, "S": args.S},
            "results": all_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
