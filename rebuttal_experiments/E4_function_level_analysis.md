# E4. Function coverage and where the branch gains come from

**Answers:** jDfu W3 and Q3 ("report function/method coverage and separate
branch coverage gains in newly reached functions from gains in previously
reached functions"); supports jDfu Q1 and 1zqK W3.

No LLM calls. Cost $0.

## Data

Stored traces in `results/repo_explore_bench/full_run_*.json` (cluster): for
each (target, strategy, seed), the ordered list of executed scripts. The
stored results hold branch COUNTS per step, not arcs, so we replay.

## Procedure: `scripts/active/analyze_function_coverage.py` (NEW)

1. For each (target, strategy, seed): start a `DockerCoverageRunner`
   (`DOCKER_MODE=exec`), replay the trace in order, and after each step dump
   the cumulative executed arcs and lines for the target module
   (`coverage json`). Check that the replayed final branch count matches the
   stored `final`; flag and exclude targets that differ by more than 2%
   (non-deterministic scripts) and report how many.
2. Map lines/arcs to functions with `covbayes.module_functions` (arc belongs
   to the function containing its source line; module-level arcs go to a
   `<module>` bucket).
3. Cyclomatic complexity per function with `radon cc` (add `radon` to the
   analysis env only).

Replay is about 24 executions x 93 targets x 4 strategies x models; with
`docker exec` mode this is about 1 h per model at `--parallel 48`.

## Metrics

Per strategy and model, mean over targets:

- **Function coverage**: reached functions / all functions (reached = at
  least one body line executed, the method's own rule).
- **Within-function branch coverage**: covered arcs / total arcs, over reached
  functions only. Total arcs per function from `coverage`'s static analysis
  (`compute_branch_totals.py` already does this per module).
- **Step-level split (jDfu's wording)**: for every step, new branches in
  functions first reached at that step ("newly reached") vs new branches in
  functions reached at an earlier step ("previously reached"). Sum over the
  run; report both totals and the share.
- **Paired decomposition of our gain over CovGreedy** at budget 24, per
  target: gain = (branches in functions only we reach) - (branches in
  functions only CovGreedy reaches) + (difference inside functions both
  reach). Report the three terms.
- **By complexity**: tertiles of cyclomatic complexity; function coverage and
  within-function branch coverage per tertile (jDfu Q1 on high-complexity
  functions).

## Output

One table (strategy x {function coverage, within-function branch coverage,
% of new branches from previously reached functions}) per model, one stacked
bar for the decomposition, one small table by complexity tertile.

## What each outcome means

- Gain mostly from newly reached functions: confirms the function-level state
  does what it claims (breadth); the depth question is then answered by E5.
- Non-trivial share from previously reached functions: the coverage map and
  plan history already drive depth inside functions, even though the scorer
  does not model it. State the share.
- Lower within-function coverage than CovGreedy in the top complexity tertile:
  say so, and point to E5's hybrid as the fix.
