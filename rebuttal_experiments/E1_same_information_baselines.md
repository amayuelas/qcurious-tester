# E1. Baselines with the same information

**Answers:** Zakb W3 and Q2 ("How would Random, Greedy, and CovGreedy perform
if provided with the same target function code snippets, target-guided
instructions, and comparable inference budgets?").

**Risk:** this is the experiment most likely to shrink our margin. Run first.

## Question

Is the gain due to the extra code context (snippets of unexecuted functions),
or to planning toward posterior-chosen targets and selecting among plans?

## Conditions

All conditions call `target_hints(source, funcs, covered_lines, K)` once per
round, exactly as the method does, and candidate i of K gets hint i (function
names with unexecuted line counts + their source snippet + the "TARGET these
functions" instruction).

| Strategy | Generation | Selection | Status |
|---|---|---|---|
| `random_tgt` | `gen_standard` + hint i | random | NEW |
| `greedy_tgt` | `gen_standard` + hint i | `select_greedy` | NEW |
| `cov_greedy_tgt` | `gen_cov_greedy` + hint i | random | NEW |
| `cov_qvalue_tgt_rnd` | targeted S=3 plans | random plan | exists |
| `covqvalue2` | targeted S=3 plans | Bayesian Q | exists |
| `cov_greedy` | untargeted | random | exists (reference) |

`cov_qvalue_tgt_rnd` vs `cov_greedy_tgt` isolates multi-step planning with the
information held equal. `covqvalue2` vs `cov_qvalue_tgt_rnd` isolates selection.

## Implementation

In `run_repo_explore_bench.py`:

1. Add `hint=None` to `gen_standard` and `gen_cov_greedy`; build one prompt per
   candidate (instead of `[prompt] * K`) and append `hints[i]` when not None.
   Use the same instruction text that `gen_k_plans` uses for targets.
2. In `run_strategy`, for strategies ending in `_tgt` among
   random/greedy/cov_greedy: compute `tgt_funcs = module_functions(source)` and
   call `target_hints(...)` each round with the touched rule (same as method).
3. Register the three names; no change to budget accounting (1 execution per
   round for these baselines, 3 per round for plans).

## Inference budget (report it, since Zakb asks for "comparable")

Per execution: method = (3 plan generations + 3 scoring calls) / 3 executions
= 2 calls; CovGreedy = 3 generations per execution = 3 calls; Greedy = 4.
So the baselines get at least as many calls. Run with `--per-strategy-cost`
and report calls, input tokens and output tokens per strategy next to coverage.

## Scale and cost

- Gemma: 93 targets x seeds 42/123/456, all six strategies. Free, ~1.5 h/seed.
- Gemini 3.8 Flash: seed 42, the five non-reference strategies. About $40.
- GLM: optional, about $60.

```bash
.venv/bin/python run_repo_explore_bench.py --seeds 42 123 456 --parallel 48 \
  --strategies cov_greedy random_tgt greedy_tgt cov_greedy_tgt \
               cov_qvalue_tgt_rnd covqvalue2 \
  --per-strategy-cost --output rebuttal_iclr27/E1_sameinfo_gemma.json \
  > results/rebuttal_iclr27/E1_sameinfo_gemma.log 2>&1
```

## Analysis

Table: mean branches (+/- SE), calls and tokens per target, for each strategy
and model. Paired tests: `covqvalue2` vs `cov_greedy_tgt`,
`cov_qvalue_tgt_rnd` vs `cov_greedy_tgt`, `cov_greedy_tgt` vs `cov_greedy`.

## Decision rule (fixed before the run)

- `covqvalue2` beats `cov_greedy_tgt` by more than 3 branches, p<0.05, on both
  models: the snippets do not explain the gain. Report the share of the
  original gap that information alone closes.
- Gap closes to within noise: the contribution is the posterior-chosen target
  information itself. Say so, move `cov_greedy_tgt` into the main table as the
  strongest baseline, and rest the planning claim on whatever
  `cov_qvalue_tgt_rnd` vs `cov_greedy_tgt` shows.
