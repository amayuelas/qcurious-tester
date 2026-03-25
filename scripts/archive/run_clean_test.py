"""Clean test of Schmidhuber's framework with all fixes.

Fixes applied:
  1. Multi-prediction uncertainty (not sampling) — ask for top-3 predictions with confidence
  2. Binary outcome branching for Q-values — success/failure, not specific output
  3. Black-box mode — no source code, must learn through interaction

Setup:
  - gpt-oss-120b for EVERYTHING (calibrated logprobs, no model mismatch)
  - Obfuscated programs (no prior knowledge)
  - 30 steps (enough to build a world model)

Strategies:
  - random: random selection
  - greedy_bb: ask which test reveals the most (no coverage info)
  - curiosity_logprob: select by logprob entropy
  - curiosity_multi_pred: select by multi-prediction diversity (Fix 1)
  - curiosity_qvalue_binary: Q-value with binary outcome branching (Fix 2)
  - oracle: execute all, pick best
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
    generate_with_model, generate_with_logprobs, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.runner.trace_parser import extract_function_signature
from curiosity_explorer.benchmarks.obfuscated_programs import load_obfuscated_programs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

MODEL = config.LOGPROB_MODEL  # gpt-oss-120b for everything

STRATEGIES = [
    "random", "greedy_bb", "curiosity_logprob",
    "curiosity_multi_pred", "curiosity_qvalue_binary", "oracle",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Schmidhuber test")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whitebox", action="store_true",
                        help="White-box mode (default is black-box)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _bb_context(func_name, source, test_history):
    """Black-box: signature + history only."""
    sig = extract_function_signature(source)
    ctx = f"Function:\n```python\n{sig}\n```\n"
    if test_history:
        ctx += "\nTest results so far:\n"
        for tc, res in test_history[-10:]:
            out = res.output or res.exception or "None"
            ctx += f"  {tc} → {out}\n"
    return ctx


def _wb_context(func_name, source, test_history):
    """White-box: full source + history."""
    ctx = f"Function:\n```python\n{source}\n```\n"
    if test_history:
        ctx += "\nTest results so far:\n"
        for tc, res in test_history[-10:]:
            out = res.output or res.exception or "None"
            ctx += f"  {tc} → {out}\n"
    return ctx


# ---------------------------------------------------------------------------
# Candidate generation (same model for everything)
# ---------------------------------------------------------------------------

def generate_candidates(func_name, source, test_history, K, ctx_fn):
    ctx = ctx_fn(func_name, source, test_history)
    prompt = f"""{ctx}

Generate a single test call that would reveal NEW behavior not seen above.
Respond with ONLY the function call. Example: {func_name}(arg1, arg2)"""

    candidates = []
    with ThreadPoolExecutor(max_workers=K) as ex:
        futures = [ex.submit(generate_with_model, MODEL, prompt, 0.9, 256)
                   for _ in range(K)]
        for f in as_completed(futures):
            try:
                resp = f.result().strip().split("\n")[0].strip()
                m = re.search(rf'{func_name}\([^)]*\)', resp)
                if m:
                    call = m.group(0)
                    if call.count("(") == call.count(")") and call not in candidates:
                        candidates.append(call)
            except:
                pass
    return candidates


# ---------------------------------------------------------------------------
# Fix 1: Multi-prediction uncertainty (one call, multiple predictions)
# ---------------------------------------------------------------------------

def score_multi_prediction(func_name, source, test_history, candidate, ctx_fn):
    """Ask model to list top-3 possible outputs with confidence."""
    ctx = ctx_fn(func_name, source, test_history)

    prompt = f"""{ctx}

For this test: {candidate}

List the 3 most likely outputs with your confidence percentage.
Format EXACTLY as:
P1: [confidence]% → [output]
P2: [confidence]% → [output]
P3: [confidence]% → [output]"""

    resp = generate_with_model(MODEL, prompt, 0.3, 200)

    # Parse confidence values
    confidences = []
    for line in resp.split("\n"):
        m = re.search(r'(\d+)%', line)
        if m:
            confidences.append(int(m.group(1)))

    if not confidences:
        return 0.0

    # Normalize to probabilities and compute entropy
    total = sum(confidences)
    if total == 0:
        return 0.0
    probs = [c / total for c in confidences]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


# ---------------------------------------------------------------------------
# Fix 2: Q-value with binary outcome branching
# ---------------------------------------------------------------------------

def score_qvalue_binary(func_name, source, test_history, candidate,
                        ctx_fn, gamma=0.5):
    """Q-value with binary branching: success vs failure paths."""
    ctx = ctx_fn(func_name, source, test_history)

    # Immediate info gain via logprobs
    predict_prompt = f"""{ctx}

What will be the output of: {candidate}

Respond with ONLY the expected output, nothing else."""

    result = generate_with_logprobs(MODEL, predict_prompt, 0.3, 100, 5)
    if not result or not result["token_logprobs"]:
        ig = 0.0
    else:
        ents = []
        for tok in result["token_logprobs"]:
            lps = tok.get("top_logprobs", {})
            if not lps:
                continue
            probs = [math.exp(lp) for lp in lps.values()]
            t = sum(probs)
            if t <= 0:
                continue
            probs = [p / t for p in probs]
            ents.append(-sum(p * math.log2(p) for p in probs if p > 0))
        ig = sum(ents) / len(ents) if ents else 0.0

    # Binary future branching
    future_value = 0.0

    for outcome_type in ["succeeds (returns a normal result)",
                         "fails (raises an error or returns an error dict)"]:
        branch_prompt = f"""{ctx}

Suppose you ran {candidate} and it {outcome_type}.
What would be the BEST next test to run to discover more behavior?
Respond with ONLY the function call."""

        future_resp = generate_with_model(MODEL, branch_prompt, 0.7, 256)
        future_call = None
        m = re.search(rf'{func_name}\([^)]*\)', future_resp)
        if m:
            future_call = m.group(0)

        if future_call:
            # Score future candidate's uncertainty
            future_predict = f"""{ctx}

(After running previous tests including {candidate})
What will be the output of: {future_call}

Respond with ONLY the expected output."""

            fr = generate_with_logprobs(MODEL, future_predict, 0.3, 100, 5)
            if fr and fr["token_logprobs"]:
                fents = []
                for tok in fr["token_logprobs"]:
                    lps = tok.get("top_logprobs", {})
                    if not lps:
                        continue
                    probs = [math.exp(lp) for lp in lps.values()]
                    t = sum(probs)
                    if t <= 0:
                        continue
                    probs = [p / t for p in probs]
                    fents.append(-sum(p * math.log2(p) for p in probs if p > 0))
                if fents:
                    future_value = max(future_value, sum(fents) / len(fents))

    return ig + gamma * future_value


# ---------------------------------------------------------------------------
# Logprob entropy scoring
# ---------------------------------------------------------------------------

def score_logprob_entropy(func_name, source, test_history, candidate, ctx_fn):
    ctx = ctx_fn(func_name, source, test_history)
    prompt = f"""{ctx}

What will be the output of: {candidate}

Respond with ONLY the expected output, nothing else."""

    result = generate_with_logprobs(MODEL, prompt, 0.3, 100, 5)
    if not result or not result["token_logprobs"]:
        return 0.0
    ents = []
    for tok in result["token_logprobs"]:
        lps = tok.get("top_logprobs", {})
        if not lps:
            continue
        probs = [math.exp(lp) for lp in lps.values()]
        t = sum(probs)
        if t <= 0:
            continue
        probs = [p / t for p in probs]
        ents.append(-sum(p * math.log2(p) for p in probs if p > 0))
    return sum(ents) / len(ents) if ents else 0.0


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(func_name, source, strategy, budget, K, seed, ctx_fn):
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []

    for step in range(budget):
        candidates = generate_candidates(func_name, source, test_history, K,
                                         ctx_fn)
        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            continue

        if strategy == "random":
            selected = random.choice(candidates)

        elif strategy == "greedy_bb":
            ctx = ctx_fn(func_name, source, test_history)
            cl = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
            prompt = f"""{ctx}

Which test is MOST LIKELY to reveal NEW behavior?
Candidates:
{cl}

Respond with ONLY the number."""
            resp = generate_with_model(MODEL, prompt, 0.3, 20)
            selected = candidates[0]
            for n in re.findall(r'\d+', resp):
                idx = int(n) - 1
                if 0 <= idx < len(candidates):
                    selected = candidates[idx]
                    break

        elif strategy == "curiosity_logprob":
            scores = {}
            with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as ex:
                futs = {ex.submit(score_logprob_entropy, func_name, source,
                                  test_history, c, ctx_fn): c
                        for c in candidates}
                for f in as_completed(futs):
                    scores[futs[f]] = f.result()
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy == "curiosity_multi_pred":
            scores = {}
            with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as ex:
                futs = {ex.submit(score_multi_prediction, func_name, source,
                                  test_history, c, ctx_fn): c
                        for c in candidates}
                for f in as_completed(futs):
                    scores[futs[f]] = f.result()
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy == "curiosity_qvalue_binary":
            scores = {}
            # Q-value is expensive, score sequentially
            for c in candidates:
                scores[c] = score_qvalue_binary(func_name, source,
                                                test_history, c, ctx_fn)
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy == "oracle":
            best, bg = candidates[0], -1
            for c in candidates:
                tr = CoverageRunner(func_name, source)
                tr.cumulative_branches = set(runner.cumulative_branches)
                r = tr.run_test(c)
                if r.new_branches > bg:
                    best, bg = c, r.new_branches
            selected = best

        else:
            selected = candidates[0]

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

    return curve


def main():
    args = parse_args()
    reset_cost()

    ctx_fn = _wb_context if args.whitebox else _bb_context
    mode = "WHITE-BOX" if args.whitebox else "BLACK-BOX"

    programs = load_obfuscated_programs()

    print("=" * 70, flush=True)
    print(f"Clean Schmidhuber Test ({mode})", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {MODEL} (for everything)", flush=True)
    print(f"  Mode: {mode}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}", flush=True)
    print(f"  Programs: {list(programs.keys())}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)
    print("=" * 70, flush=True)

    test = generate_with_model(MODEL, "Say ok", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]

        print(f"\n[{i+1}/{len(programs)}] {key} ({func_name})", flush=True)

        func_results = {"key": key, "func_name": func_name, "strategies": {}}

        # Run strategies in parallel
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, s,
                                args.budget, args.K, args.seed, ctx_fn): s
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

    # Results
    def mean(v):
        return sum(v) / len(v) if v else 0

    print(f"\n{'=' * 70}", flush=True)
    print(f"RESULTS ({mode})", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n  {'Strategy':<25} {'Mean Final':>10} {'Mean @10':>10} {'Mean @20':>10}",
          flush=True)
    print(f"  {'─' * 57}", flush=True)
    for s in STRATEGIES:
        finals = [r["strategies"][s][-1] for r in all_results if r["strategies"][s]]
        at10 = [r["strategies"][s][9] for r in all_results
                if r["strategies"][s] and len(r["strategies"][s]) >= 10]
        at20 = [r["strategies"][s][19] for r in all_results
                if r["strategies"][s] and len(r["strategies"][s]) >= 20]
        print(f"  {s:<25} {mean(finals):>10.1f} {mean(at10):>10.1f} "
              f"{mean(at20):>10.1f}", flush=True)

    # Per-function
    print(f"\n  {'Function':<12}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s[:10]:>11}", end="")
    print(flush=True)
    for r in all_results:
        print(f"  {r['key']:<12}", end="")
        for s in STRATEGIES:
            v = r["strategies"][s][-1] if r["strategies"][s] else 0
            print(f" {v:>11}", end="")
        print(flush=True)

    # Step-by-step for each function
    for r in all_results:
        print(f"\n  {r['key']} step-by-step:", flush=True)
        print(f"  {'Step':>5}", end="", flush=True)
        for s in STRATEGIES:
            print(f" {s[:10]:>11}", end="")
        print(flush=True)
        budget = args.budget
        for step in range(0, budget, 5):
            print(f"  {step+1:>5}", end="")
            for s in STRATEGIES:
                v = r["strategies"][s]
                print(f" {v[step] if step < len(v) else 0:>11}", end="")
            print(flush=True)
        # Final
        print(f"  {budget:>5}", end="")
        for s in STRATEGIES:
            v = r["strategies"][s]
            print(f" {v[-1] if v else 0:>11}", end="")
        print(flush=True)

    # Win rates
    print(f"\n  Win rates vs random:", flush=True)
    for s in STRATEGIES:
        if s == "random":
            continue
        w = l = t = 0
        for r in all_results:
            a = r["strategies"][s][-1] if r["strategies"][s] else 0
            b = r["strategies"]["random"][-1] if r["strategies"]["random"] else 0
            if a > b: w += 1
            elif a < b: l += 1
            else: t += 1
        n = w + l + t
        if n:
            print(f"    {s:<25} W={w} L={l} T={t}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "wb" if args.whitebox else "bb"
    with open(config.RESULTS_DIR / f"clean_test_{tag}.json", "w") as f:
        json.dump({"model": MODEL, "mode": mode, "budget": args.budget,
                    "K": args.K, "results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"Saved", flush=True)


if __name__ == "__main__":
    main()
