"""Test curiosity Q-values against baselines.

Strategies:
  - random: Random selection
  - greedy: Ask LLM for best coverage candidate
  - curiosity_greedy: Select by immediate info gain only (γ=0)
  - curiosity_qvalue: Select by Q-value with lookahead (γ=0.5)
  - curiosity_qvalue_high: Q-value with stronger planning (γ=0.7)
  - oracle: Single-step greedy oracle

Usage:
    python run_qvalue_test.py --toy --budget 15 --K 5
    python run_qvalue_test.py --targeted --budget 15 --K 5
"""

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import generate_with_model, get_cost, reset_cost
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.info_gain import (
    _build_context, estimate_output_entropy,
)
from curiosity_explorer.explorer.q_values import compute_q_values

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
    "random", "greedy", "curiosity_greedy",
    "curiosity_qvalue", "curiosity_qvalue_high", "oracle",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Q-value test")
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


def _select_greedy(candidates, func_name, source, test_history, runner):
    code_section, history_str = _build_context(func_name, source, test_history)
    if test_history:
        recent = test_history[-5:]
        history_str = "Previous tests and coverage results:\n"
        for tc, res in recent:
            history_str += (f"  {tc} → new_branches={res.new_branches}, "
                           f"output={res.output or res.exception}\n")
    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (f"Given this function:\n{code_section}\n\n{history_str}\n"
              f"Total branches covered: {runner.get_cumulative_coverage()}\n\n"
              f"Which test covers the most NEW branches? "
              f"Respond with ONLY the number.\n\nCandidates:\n{cand_list}")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _select_oracle(candidates, func_name, source, runner):
    best, best_gain = candidates[0], -1
    for c in candidates:
        tr = CoverageRunner(func_name, source)
        tr.cumulative_branches = set(runner.cumulative_branches)
        r = tr.run_test(c)
        if r.new_branches > best_gain:
            best, best_gain = c, r.new_branches
    return best


def run_strategy(func_name, source, strategy, budget, K, S, seed):
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []
    diagnostics = []

    for step in range(budget):
        candidates = generate_test_candidates(
            func_name, source, test_history=test_history, K=K
        )
        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            continue

        diag = {"step": step}

        if strategy == "random":
            selected = random.choice(candidates)

        elif strategy == "greedy":
            selected = _select_greedy(candidates, func_name, source,
                                      test_history, runner)

        elif strategy == "curiosity_greedy":
            # γ=0: immediate info gain only
            scores = {}
            for c in candidates:
                scores[c] = estimate_output_entropy(
                    func_name, source, test_history, c, S=S
                )
            selected = max(candidates, key=lambda c: scores.get(c, 0))
            diag["scores"] = {c: scores[c] for c in candidates}

        elif strategy in ("curiosity_qvalue", "curiosity_qvalue_high"):
            gamma = 0.5 if strategy == "curiosity_qvalue" else 0.7
            q_results = compute_q_values(
                func_name, source, test_history, candidates,
                gamma=gamma, S=S, future_K=3,
            )
            selected = max(candidates,
                           key=lambda c: q_results.get(c, {}).get("q_value", 0))
            diag["q_results"] = q_results

        elif strategy == "oracle":
            selected = _select_oracle(candidates, func_name, source, runner)

        else:
            selected = candidates[0]

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

        diag["selected"] = selected
        diag["new_branches"] = result.new_branches
        diag["cumulative"] = runner.get_cumulative_coverage()
        diagnostics.append(diag)

    return curve, diagnostics


def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("Curiosity Q-Value Test", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)

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
            programs = dict(corridor)
            programs.update({k: non_all[k] for k in nc_keys[:needed]})

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
            "diagnostics": {},
        }

        # Run all strategies in parallel
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, s,
                                args.budget, args.K, args.S, args.seed): s
                for s in STRATEGIES
            }
            for future in as_completed(futures):
                s = futures[future]
                try:
                    curve, diag = future.result()
                except Exception as e:
                    log.warning(f"{s} failed: {e}")
                    curve, diag = [], []
                func_results["strategies"][s] = curve
                func_results["diagnostics"][s] = diag
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
    print("WIN RATES", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cmp in ["curiosity_qvalue", "curiosity_qvalue_high", "curiosity_greedy"]:
        for base in ["greedy", "random", "oracle"]:
            w = l = t = 0
            for r in all_results:
                a = r["strategies"][cmp][-1] if r["strategies"][cmp] else 0
                b = r["strategies"][base][-1] if r["strategies"][base] else 0
                if a > b: w += 1
                elif a < b: l += 1
                else: t += 1
            n = w + l + t
            if n:
                print(f"  {cmp:<25} vs {base:<10} W={w:>2} L={l:>2} T={t:>2} "
                      f"(win={w/n:.0%})", flush=True)

    # Per-function
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

    # Q-value diagnostics for corridor functions
    print(f"\n{'=' * 70}", flush=True)
    print("Q-VALUE DIAGNOSTICS (corridor functions)", flush=True)
    print(f"{'=' * 70}", flush=True)
    for r in all_results:
        if not r["is_corridor"]:
            continue
        qdiag = r["diagnostics"].get("curiosity_qvalue", [])
        if not qdiag:
            continue
        print(f"\n  {r['func_name']}:", flush=True)
        for d in qdiag[:5]:  # first 5 steps
            qr = d.get("q_results", {})
            sel = d.get("selected", "?")
            nb = d.get("new_branches", 0)
            cum = d.get("cumulative", 0)
            # Show the selected candidate's Q breakdown
            if sel in qr:
                q = qr[sel]
                print(f"    Step {d['step']+1}: {sel[:50]}", flush=True)
                print(f"      q={q['q_value']:.2f} = ig={q['immediate_ig']:.2f} "
                      f"+ γ*fv={q['future_value']:.2f} → "
                      f"new_branches={nb}, cumulative={cum}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "toy" if args.toy else ("targeted" if args.targeted else "ult")
    out_path = config.RESULTS_DIR / f"qvalue_{tag}.json"
    # Don't save full diagnostics (too large) — save summary only
    save_results = []
    for r in all_results:
        save_results.append({
            "func_key": r["func_key"], "func_name": r["func_name"],
            "is_corridor": r["is_corridor"], "strategies": r["strategies"],
        })
    with open(out_path, "w") as f:
        json.dump({
            "config": {"model": config.MODEL, "budget": args.budget,
                       "K": args.K, "S": args.S, "strategies": STRATEGIES},
            "results": save_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
