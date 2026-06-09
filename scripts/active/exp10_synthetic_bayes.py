"""Experiment 10 — Synthetic corridor benchmark with a computable optimum.

Addresses the universal "Bayesian framing is analogy, not method" critique by
constructing a setting where the optimal exploration policy is *computable*, then
showing that a greedy (immediate-gain) policy plateaus as corridor depth grows
— Sun et al. (2011)'s corridor result — while a planning policy that invests in
zero-gain setup steps reaches the deep payoff.

Synthetic program model (abstract, deterministic structure):
  - A CORRIDOR of depth d: a chain of setup "functions" c_0 -> c_1 -> ... -> c_d.
    c_0 is callable from the start; calling c_i (i<d) unlocks c_{i+1} and yields
    ZERO immediate branches (pure setup). Calling the terminal c_d yields k
    branches (the deep payoff), and is only callable once the whole corridor is
    traversed.
  - m DISTRACTORS x_1..x_m: callable from the start, each holds `distractor_cov`
    branches discovered one-per-visit (so there is always a small positive
    immediate gain available — this is what tempts a greedy policy away from the
    zero-gain corridor).

Budget N actions. Goal: maximize cumulative branches discovered.

Policies:
  - random   : uniform over currently-callable functions.
  - greedy   : argmax expected IMMEDIATE branch gain (ties broken randomly).
               Distractors (+1) always beat setup steps (0), so greedy never
               invests in the corridor -> plateaus, never reaching the payoff.
  - planner  : optimal policy WITH KNOWN STRUCTURE (a computable ceiling):
               traverse the corridor (d setup steps) then harvest the terminal,
               spending any remaining budget on distractors; vs. just harvesting
               distractors — take whichever yields more in N steps. This is the
               exact optimum for the known structure.

(The belief-space Bayes-optimal DP under structural *uncertainty*, and the LLM
CovQValue condition, are layered on top of this core separately.)

This core is fully offline/deterministic — no API/Docker.

Usage:
    python scripts/active/exp10_synthetic_bayes.py
    python scripts/active/exp10_synthetic_bayes.py --depths 1 2 3 5 8 --k 20 --budget 24
"""

import argparse
import json
import random
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


class CorridorProgram:
    """Deterministic corridor + distractors. Tracks covered branches + unlocks."""

    def __init__(self, depth, k, m, distractor_cov=2):
        self.depth = depth          # corridor length (setup steps before payoff)
        self.k = k                  # branches at the terminal
        self.m = m                  # number of distractors
        self.distractor_cov = distractor_cov
        self.reset()

    def reset(self):
        # corridor progress: how many setup steps taken (0..depth). The terminal
        # (payoff) is callable once progress == depth.
        self.progress = 0
        self.terminal_done = False
        self.distractor_seen = [0] * self.m  # branches already pulled per distractor
        self.covered = 0

    def actions(self):
        """Currently-callable actions: ('setup',), ('terminal',), ('distractor', i)."""
        acts = []
        if self.progress < self.depth:
            acts.append(("setup",))
        elif not self.terminal_done:
            acts.append(("terminal",))
        for i in range(self.m):
            if self.distractor_seen[i] < self.distractor_cov:
                acts.append(("distractor", i))
        return acts

    def immediate_gain(self, action):
        """Branches this action would reveal right now (what greedy sees)."""
        if action[0] == "setup":
            return 0
        if action[0] == "terminal":
            return self.k
        if action[0] == "distractor":
            return 1 if self.distractor_seen[action[1]] < self.distractor_cov else 0
        return 0

    def step(self, action):
        gain = self.immediate_gain(action)
        if action[0] == "setup":
            self.progress += 1
        elif action[0] == "terminal":
            self.terminal_done = True
        elif action[0] == "distractor":
            self.distractor_seen[action[1]] += 1
        self.covered += gain
        return gain


def run_random(prog, budget, rng):
    prog.reset()
    for _ in range(budget):
        acts = prog.actions()
        if not acts:
            break
        prog.step(rng.choice(acts))
    return prog.covered


def run_greedy(prog, budget, rng):
    prog.reset()
    for _ in range(budget):
        acts = prog.actions()
        if not acts:
            break
        best = max(prog.immediate_gain(a) for a in acts)
        prog.step(rng.choice([a for a in acts if prog.immediate_gain(a) == best]))
    return prog.covered


def optimal_known_structure(depth, k, m, distractor_cov, budget):
    """Exact optimum given known structure (closed form).

    Two candidate plans; take the better:
      (A) harvest distractors only: min(budget, m*distractor_cov) branches.
      (B) traverse corridor (depth setup steps, 0 each) + terminal (k) if budget
          allows, then spend the rest on distractors.
    """
    only_distract = min(budget, m * distractor_cov)
    corridor_plan = 0
    if budget >= depth + 1:                 # room to traverse + harvest terminal
        corridor_plan = k + min(budget - (depth + 1), m * distractor_cov)
    return max(only_distract, corridor_plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--k", type=int, default=50, help="terminal payoff branches")
    ap.add_argument("--m", type=int, default=30,
                    help="number of distractors (kept > budget so a greedy "
                         "policy always has a positive-gain distractor and "
                         "never has to fall into the zero-gain corridor)")
    ap.add_argument("--distractor-cov", type=int, default=1)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--output", default="repo_explore_bench/exp10_synthetic.json")
    args = ap.parse_args()

    print(f"Synthetic corridor: k={args.k} terminal, m={args.m} distractors "
          f"(cov {args.distractor_cov}), budget N={args.budget}, "
          f"{args.seeds} seeds\n")
    print(f"  {'depth':>5} {'random':>9} {'greedy':>9} {'optimal':>9} "
          f"{'opt-greedy':>11}")

    out = {"config": vars(args), "by_depth": {}}
    for d in args.depths:
        prog = CorridorProgram(d, args.k, args.m, args.distractor_cov)
        rnds, grds = [], []
        for s in range(args.seeds):
            rng = random.Random(1000 + s)
            rnds.append(run_random(prog, args.budget, rng))
            grds.append(run_greedy(prog, args.budget, rng))
        opt = optimal_known_structure(d, args.k, args.m, args.distractor_cov,
                                      args.budget)
        mr, mg = statistics.mean(rnds), statistics.mean(grds)
        gap = opt - mg
        print(f"  {d:>5} {mr:>9.1f} {mg:>9.1f} {opt:>9.1f} {gap:>10.1f}")
        out["by_depth"][d] = {"random": mr, "greedy": mg, "optimal": opt,
                              "optimal_minus_greedy": gap}

    print("\nKey: greedy plateaus FLAT (it always has a positive-gain distractor, "
          "so it never invests in the zero-gain corridor and never reaches the\n"
          "depth-d payoff) — regardless of depth. The optimum traverses the "
          "corridor and harvests the payoff, a large persistent gap. As the\n"
          "corridor deepens, the optimum's reachable payoff declines (more budget "
          "spent on setup) and only converges to greedy's plateau once the\n"
          "corridor no longer fits in the budget (d >= N).")

    outpath = RESULTS_DIR / args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
