# E5. Branch-aware targeting

**Answers:** jDfu W1 and Q1; 1zqK W3 and Q3 ("How does CovQValue account for
uncovered branches within previously executed functions?").

## Question

Under the touched rule a function leaves the target set once any body line
runs. Does keeping partly covered functions as targets, weighted by what is
left in them, improve branch coverage?

## Conditions (valuation fixed to the E3 winner; until then use `fnt_bayes`)

| Strategy | Target set | Weight |
|---|---|---|
| `cov_qvalue_tgt_fnt_bayes` | untouched functions (current) | lines |
| `cov_qvalue_tgt_rem_fnt_bayes` | any function with unexecuted lines (exists) | remaining lines |
| `cov_qvalue_tgt_hyb_fnt_bayes` (NEW) | untouched + revisit pool | see below |

The 27-target pilot found `_rem` 16 branches worse than touched on real code
(p=0.015): it keeps re-targeting large, mostly covered functions. The hybrid
is designed to avoid exactly that.

## Hybrid design (`_hyb`)

- Per function, track uncovered branch arcs: static arcs of the function
  (from `coverage`'s analysis) minus executed arcs. Requires the runner to
  expose cumulative arcs per function (extend `remaining_by_function` to
  return arcs as well as lines).
- Each round, K target chunks:
  - K-1 chunks from untouched functions, as now.
  - 1 revisit chunk: touched functions ranked by uncovered arcs, top 6,
    excluding functions with fewer than 2 uncovered arcs and functions that
    were in the revisit chunk in the last 2 rounds without gaining an arc
    (stops repeated attempts at dead or unreachable branches).
  - If no untouched functions remain, all K chunks are revisit chunks.
- Revisit hint text: name the function, show its snippet, and list the
  uncovered branch lines ("the branch at line N has never been taken"), so the
  generator knows which condition to flip.
- Scorer listing and Q use uncovered arcs as the size for revisit functions,
  so both kinds of target are valued in the unit we evaluate (branches).

## Scale and cost

Gemma, 93 targets, seeds 42/123/456, three strategies in one job. Free, about
1 h per seed. Plus hidden-key corridors depths 4/6/8 (the corridor needs
partly covered stage functions to stay targeted).

```bash
.venv/bin/python run_repo_explore_bench.py --seeds 42 123 456 --parallel 48 \
  --strategies cov_qvalue_tgt_fnt_bayes cov_qvalue_tgt_rem_fnt_bayes \
               cov_qvalue_tgt_hyb_fnt_bayes \
  --output rebuttal_iclr27/E5_targeting_gemma.json
```

Also run at budget 48 for one seed: with a longer budget the untouched pool
empties and depth is all that is left, so the touched rule should saturate
first if the reviewers are right.

## Analysis

Mean branches, paired tests, and the E4 metrics for each variant (function
coverage vs within-function branch coverage), plus the complexity-tertile
breakdown. Coverage curves over the budget for the three variants.

## Decision rule

- Hybrid beats touched by more than 3 branches (p<0.05) at budget 24, or
  clearly at budget 48 without losing at 24: adopt it; Algorithm 1 is
  rewritten with a branch-level revisit set, which closes this weakness.
- No difference: keep the touched rule and report all three, with the E4
  split showing how much depth the method already gets.
- Hybrid worse: report it as evidence that at this budget breadth is the
  better use of executions, and show the budget-48 curve.
