"""Evaluate plan scorers offline on logged candidate pools.

Input: a cov_qvalue*_calib run made after candidate plans were logged (each
calib round stores every candidate's full plan, its realized gain from a
trial run, and the scorer inputs at decision time: coverage summary and the
uncovered functions). Every scorer below re-scores exactly the same pools, so
differences are selection quality alone, with no execution noise.

Scorers:
  current  production prompt (_score_plans: plan steps cut to 200 chars)
  A        same prompt, full plan steps
  B        A + the uncovered-function list (with sizes) in the prompt
  B2       LLM lists which uncovered functions a plan executes; score = their
           total size (lines)
  C        static, no LLM: uncovered functions the plan's code references by
           name, weighted by size
  C+A      C, ties broken by A's Q
  D        one comparative call: all K full plans side by side, pick the best

Metrics (ties among top scores broken uniformly at random, in expectation):
  top-1 accuracy on decisive rounds (unique realized best) vs 1/K chance,
  and share of the pool-mean -> oracle gap closed by the pick.

Usage:
    MODEL=gemma-4-31B-it python scripts/active/eval_scorers_offline.py \
        --input results/repo_explore_bench/pilot_scorerpool_gemma.json
"""

import argparse
import ast
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import run_repo_explore_bench as R  # noqa: E402
from curiosity_explorer.llm import generate_with_model, get_cost  # noqa: E402

GAMMA = 0.5


class _LoggedMap:
    """Stands in for CoverageMap: returns the summary logged for the round."""
    def __init__(self, summary):
        self._summary = summary
        self.score_map_mode = None

    def coverage_summary(self, mode=None):
        return self._summary


def _size(f):
    _q, start, end = f
    return max(end - start + 1, 1)


def _plan_text(plan):
    return "".join(f"\nStep {i+1}:\n```python\n{s}\n```\n" for i, s in enumerate(plan))


def _uncovered_text(uncovered, limit=40):
    items = sorted(uncovered, key=lambda f: -_size(f))[:limit]
    return "\n".join(f"  {q} ({_size(f)} lines)" for f in items for q in [f[0]])


def _llm_q(prompt):
    resp = generate_with_model(config.MODEL, prompt, 0.3, 256)
    nums = re.findall(r"\d+", resp)
    imm = int(nums[0]) if nums else 0
    fut = int(nums[1]) if len(nums) > 1 else 0
    return imm + GAMMA * fut


# --- scorers: each returns a list of K scores for one round -----------------

def score_current(rd, module, source):
    s = R._score_plans([c["plan"] for c in rd["candidates"]], module, source,
                       _LoggedMap(rd["cov_summary"]), GAMMA)
    return [s[i]["q"] for i in range(len(rd["candidates"]))]


QA_TEMPLATE = """Module: {module}
```python
{code}
```

{cov_summary}
{extra}
Consider this TEST PLAN (a sequence of {n} scripts):
{plan}

Evaluate by answering TWO questions with just numbers:
1. IMMEDIATE GAIN: Total NEW branches this plan discovers? (0-50)
2. FUTURE VALUE: Additional branches reachable AFTER this plan? (0-50)

Format: immediate, future
Example: 15, 25"""


def _score_qa(rd, module, source, extra=""):
    def one(c):
        return _llm_q(QA_TEMPLATE.format(
            module=module, code=source[:2000], cov_summary=rd["cov_summary"],
            extra=extra, n=len(c["plan"]), plan=_plan_text(c["plan"])))
    with ThreadPoolExecutor(len(rd["candidates"])) as ex:
        return list(ex.map(one, rd["candidates"]))


def score_A(rd, module, source):
    return _score_qa(rd, module, source)


def score_B(rd, module, source):
    extra = ("\nFUNCTIONS NO TEST HAS EXECUTED YET (size in lines):\n"
             f"{_uncovered_text(rd['uncovered'])}\n"
             "New branches come mostly from executing these.\n")
    return _score_qa(rd, module, source, extra)


def score_B2(rd, module, source):
    by_name = {f[0]: f for f in rd["uncovered"]}

    def one(c):
        prompt = f"""Module: {module}

FUNCTIONS NO TEST HAS EXECUTED YET:
{_uncovered_text(rd['uncovered'], limit=80)}

TEST PLAN (scripts run in order):
{_plan_text(c['plan'])}

Which of the listed functions will this plan actually execute (directly or
indirectly)? Answer with ONLY the function names from the list, one per line.
If none, answer NONE."""
        resp = generate_with_model(config.MODEL, prompt, 0.3, 512)
        hit = {n for n in by_name if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", resp)}
        return sum(_size(by_name[n]) for n in hit)
    with ThreadPoolExecutor(len(rd["candidates"])) as ex:
        return list(ex.map(one, rd["candidates"]))


def _referenced_names(plan):
    names = set()
    for script in plan:
        try:
            tree = ast.parse(script)
        except SyntaxError:
            names.update(re.findall(r"[A-Za-z_]\w*", script))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[-1])
                if node.asname:
                    names.add(node.asname)
    return names


def score_C(rd, module, source):
    out = []
    for c in rd["candidates"]:
        names = _referenced_names(c["plan"])
        score = 0
        for f in rd["uncovered"]:
            parts = f[0].split(".")
            fn = parts[-1]
            # dunders (__init__, __repr__...) are triggered by using the class
            key = parts[-2] if fn.startswith("__") and len(parts) > 1 else fn
            if key in names:
                score += _size(f)
        out.append(score)
    return out


def score_CA(rd, module, source, cache):
    c = score_C(rd, module, source)
    a = cache["A"]
    return [ci * 1000 + ai for ci, ai in zip(c, a)]


def score_D(rd, module, source):
    K = len(rd["candidates"])
    plans = "\n".join(f"=== PLAN {i} ==={_plan_text(c['plan'])}"
                      for i, c in enumerate(rd["candidates"]))
    prompt = f"""Module: {module}
```python
{source[:2000]}
```

{rd['cov_summary']}

FUNCTIONS NO TEST HAS EXECUTED YET (size in lines):
{_uncovered_text(rd['uncovered'])}

Here are {K} candidate test plans:
{plans}

Which ONE plan will discover the most NEW branches when executed? Consider
which uncovered functions each plan really reaches and whether its setup works.
Answer with ONLY the plan number (0-{K-1})."""
    resp = generate_with_model(config.MODEL, prompt, 0.3, 64)
    m = re.search(r"\d+", resp)
    best = int(m.group()) if m and int(m.group()) < K else None
    return [1 if i == best else 0 for i in range(K)]


# --- metrics ----------------------------------------------------------------

def evaluate(rounds, scores):
    hit = n_dec = 0.0
    pick = pool = oracle = 0.0
    for rd, sc in zip(rounds, scores):
        real = [c["realized_gain"] for c in rd["candidates"]]
        top = max(sc)
        ties = [i for i, s in enumerate(sc) if s == top]
        exp_pick = statistics.mean(real[i] for i in ties)
        pick += exp_pick
        pool += statistics.mean(real)
        oracle += max(real)
        if real.count(max(real)) == 1:
            n_dec += 1
            best = real.index(max(real))
            hit += (best in ties) / len(ties)
    gap = oracle - pool
    return {"top1": hit / n_dec if n_dec else None, "n_decisive": int(n_dec),
            "gap_closed": (pick - pool) / gap if gap else None,
            "picked_per_round": pick / len(rounds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--strategy", default="cov_qvalue_tgt_calib")
    ap.add_argument("--scorers", nargs="+",
                    default=["current", "A", "B", "B2", "C", "C+A", "D"])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    data = json.load(open(ROOT / args.input))
    rounds, meta = [], []
    for r in data["results"]:
        for rd in r["strategies"].get(args.strategy, {}).get("calib", []):
            if len(rd["candidates"]) >= 2 and "plan" in rd["candidates"][0]:
                rounds.append(rd)
                meta.append(r["module"])
    modules = sorted(set(meta))
    with ThreadPoolExecutor(8) as ex:
        sources = dict(zip(modules, ex.map(R.fetch_source, modules)))
    print(f"{len(rounds)} rounds from {len(modules)} targets "
          f"(model {config.MODEL})\n")

    fns = {"current": score_current, "A": score_A, "B": score_B,
           "B2": score_B2, "C": score_C, "D": score_D}
    results, cache = {}, {}
    for name in args.scorers:
        if name == "C+A":
            if "A" not in cache:
                continue
            sc = [score_CA(rd, m, sources[m], {"A": a})
                  for rd, m, a in zip(rounds, meta, cache["A"])]
        else:
            with ThreadPoolExecutor(args.workers) as ex:
                sc = list(ex.map(lambda x: fns[name](x[0], x[1], sources[x[1]]),
                                 zip(rounds, meta)))
        cache[name] = sc
        results[name] = evaluate(rounds, sc)
        m = results[name]
        print(f"  {name:<8} top-1 {m['top1']:.1%} (n={m['n_decisive']})  "
              f"gap closed {m['gap_closed']:+.0%}  "
              f"picked {m['picked_per_round']:.2f}/round", flush=True)

    real = [[c["realized_gain"] for c in rd["candidates"]] for rd in rounds]
    print(f"\n  pool mean {statistics.mean(statistics.mean(x) for x in real):.2f}/round, "
          f"oracle {statistics.mean(max(x) for x in real):.2f}/round, "
          f"chance top-1 {statistics.mean(1/len(x) for x in real):.1%}")
    print(f"  LLM cost: {get_cost()['api_calls']} calls")
    out = ROOT / args.input.replace(".json", "_scorers.json")
    json.dump({"scorers": results, "n_rounds": len(rounds)}, open(out, "w"), indent=2)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
