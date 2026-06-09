"""CovBayes (Exp 11): principled Bayesian selection over a per-function posterior.

Replaces CovQValue's LLM-as-judge scoring with an explicit probabilistic model:

  - Per-FUNCTION reachability posterior Beta(alpha, beta): belief that a given
    function/method is coverable. Covered functions are resolved (alpha large);
    uncovered ones start at Beta(1,1).
  - The LLM only (a) generates the K diverse plans (reused from the existing
    pipeline) and (b) predicts which functions a plan will exercise, with a
    confidence q_f. It does NOT score — that's where it's weak (Exp 2).
  - Plans are ranked by CLOSED-FORM expected information gain over the uncovered
    functions: IG(a) = sum_f [ H(p_f) - q_f*H(p_f|hit) - (1-q_f)*H(p_f|miss) ].
  - After executing the chosen plan, the posterior is conjugate-updated: newly
    covered functions get alpha+=1; functions the plan was predicted to cover but
    didn't get beta+=1.

Granularity is per-function (not per coverage.py arc) so the LLM's predictions
are meaningful and the IG stays closed-form. Coverage is mapped to functions via
the module AST's line ranges.
"""

import ast
import math
import re

import config
from ..llm import generate_with_model


def module_functions(source: str):
    """[(qualname, start_line, end_line)] for top-level funcs + methods."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out, cls = [], []

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                cls.append(child.name)
                walk(child)
                cls.pop()
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join(cls + [child.name]) if cls else child.name
                end = getattr(child, "end_lineno", child.lineno)
                # Use the BODY start, not the def line: `def` lines execute at
                # import (function definition), so counting them would mark every
                # function "covered" the moment the module is imported. A function
                # is only "exercised" if a body line runs (i.e. it was called).
                # Skip a leading docstring so the threshold is real code.
                body = child.body
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and len(body) > 1:
                    body_start = body[1].lineno
                elif body:
                    body_start = body[0].lineno
                else:
                    body_start = child.lineno
                out.append((qual, body_start, end))
                # don't descend into nested defs — method/function granularity
    walk(tree)
    return out


def covered_quals(covered_lines, funcs):
    """Set of function qualnames with at least one covered line."""
    lines = {ln for (_f, ln) in covered_lines}  # module is the only --source file
    cov = set()
    for qual, s, e in funcs:
        if any(s <= ln <= e for ln in lines):
            cov.add(qual)
    return cov


def plan_expected_gain(post, uncovered, q):
    """Expected new coverage of a plan under the per-function posterior:

        score(a) = sum_{f uncovered} q_f(a) * p_f,    p_f = alpha_f/(alpha_f+beta_f)

    i.e. the immediate information-gain term ḡ(a|h) computed in closed form from
    the posterior and the LLM's branch-hit predictions q_f — replacing the
    LLM-judge's guessed number. (The pure entropy-reduction IG is degenerate at a
    uniform Beta(1,1) prior — a single hit/miss reduces entropy equally
    regardless of q — so expected coverage under the posterior is the
    discriminative, coverage-appropriate objective.) The posterior supplies the
    Bayesian feedback: functions repeatedly predicted-but-missed accrue beta, so
    p_f falls and the selector stops chasing them.

    post: {qual: [alpha, beta]}; uncovered: iterable of quals; q: {qual: prob}.
    """
    score = 0.0
    for f in uncovered:
        a, b = post[f]
        score += q.get(f, 0.03) * (a / (a + b))
    return score


def predict_plan_functions(module, source, plan, uncovered_names, model=None):
    """Ask the LLM which uncovered functions a plan will exercise (q_f).

    Returns {matched_qual: confidence}. Predicted short names are matched to
    qualnames by last component.
    """
    model = model or config.MODEL
    plan_str = "".join(f"\n# step {i+1}\n{s[:220]}" for i, s in enumerate(plan))
    shortlist = sorted(uncovered_names)[:50]
    prompt = f"""Module: {module}
```python
{source[:2000]}
```

These functions are NOT yet covered:
{', '.join(shortlist)}

Given this test plan:
{plan_str}

Which of the not-yet-covered functions will this plan actually EXERCISE (cause
to execute)? For each, give a confidence 0.0-1.0. One per line as `name: conf`.
Only list functions you expect to run; omit the rest."""
    resp = generate_with_model(model, prompt, 0.3, 400)

    # map last-component -> set of quals (for matching short predictions)
    by_short = {}
    for qual in uncovered_names:
        by_short.setdefault(qual.split(".")[-1], []).append(qual)

    q = {}
    for line in resp.splitlines():
        m = re.match(r"\s*[-*]?\s*([A-Za-z_][\w\.]*)\s*[:=]\s*([01]?\.?\d+)", line)
        if not m:
            continue
        name, conf = m.group(1), float(m.group(2))
        conf = min(max(conf, 0.0), 1.0)
        if name in uncovered_names:                     # exact qual
            q[name] = max(q.get(name, 0.0), conf)
        else:                                            # short name -> quals
            for qual in by_short.get(name.split(".")[-1], []):
                q[qual] = max(q.get(qual, 0.0), conf)
    return q
