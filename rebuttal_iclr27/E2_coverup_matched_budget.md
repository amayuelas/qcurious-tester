# E2. CoverUp at a matched budget

**Answers:** 1zqK W4 and Q4 ("How does CovQValue compare with CoverUp under
matched execution and inference budgets?").

## Question

The paper runs CoverUp to completion (Table 3: 98.0 / 45.2 / 59.8 vs ours
103.5 / 85.5 / 101.6). Does the ordering hold when
CoverUp gets the same number of test executions and LLM calls as our method?

## Step 0: measure CoverUp's native usage (free, do first)

We do not currently log what CoverUp spends. Extend
`scripts/coverup_inject/sitecustomize.py` (it already wraps `litellm.acreate`)
to write per target: number of LLM calls, prompt/completion tokens, and number
of test executions. Rerun CoverUp on Gemma (free) with logging only. This
tells us which direction matching moves the comparison.

## Conditions

| Condition | Executions | LLM calls |
|---|---|---|
| CoverUp native (existing numbers, + usage log) | uncapped | uncapped |
| CoverUp matched (NEW) | <= 24 | <= 48 |
| `covqvalue2` (from E1, same code version) | 24 | 48 (8 rounds x 6) |

## Implementation

- Env vars `COVERUP_MAX_CALLS`, `COVERUP_MAX_EXECS` read by `sitecustomize.py`.
- Call cap: after N `acreate` calls, raise a sentinel exception that CoverUp
  treats as a failed request, so it stops prompting and writes the tests it
  has. Verify on 3 targets that tests generated before the cap are kept.
- Execution cap: count CoverUp's test-run subprocess invocations (wrap the
  function CoverUp uses to run a candidate test; find it in the installed
  version and note the name here). After the cap, same stop behaviour.
- Final coverage is measured exactly as now (`coverage run --branch
  --source=<module>` over the generated tests), so numbers stay comparable.
- CoverUp works per code segment and may spend the cap on the first segments
  in file order. That is its native behaviour; do not reorder. Note it in the
  caption.

## Scale and cost

- Gemma: 93 targets, free.
- Gemini 3.8 Flash: 93 targets, capped at 48 calls per target, about $20.
- GLM: optional (needs the 1024-token thinking budget setting), about $45.

```bash
COVERUP_MAX_CALLS=48 COVERUP_MAX_EXECS=24 \
.venv/bin/python run_external_baselines.py --baseline coverup \
  --output rebuttal_iclr27/E2_coverup_matched_gemma.json \
  > results/rebuttal_iclr27/E2_coverup_matched_gemma.log 2>&1
```

## Analysis

Table per model: branches, executions, calls, tokens for the three conditions.
Paired test `covqvalue2` vs CoverUp matched. If Step 0 shows CoverUp natively
uses MORE than our budget, also report our existing budget-32 ablation point so
the reader sees both directions.

## Decision rule

Report the matched number as the primary comparison in the revised table,
whichever way it goes. If CoverUp matched is within noise of ours on Gemini,
state that on the strongest model the two are comparable and the advantage is
on the weaker models (Gemma, GLM), where the existing gaps are 40+ branches.
