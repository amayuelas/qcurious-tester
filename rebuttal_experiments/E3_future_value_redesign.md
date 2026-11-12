# E3. Redesigned future value and posteriors

**Answers:** Zakb W1, W2, Q1; 1zqK W1, W2, Q1, Q2. Also jDfu's summary remark
that most of the gain comes from targeting.

## Why the current term is flat (diagnosis, goes in the rebuttal)

`_rescore_with_posterior` computes

    g(a) = sum_{f in EXECUTES(a)} p_f * size_f
    v(a) = sum_{f uncovered, f not in EXECUTES(a)} p'_f * size_f
           p'_f = p_f + (1 - p_f) * 0.09   if f in ENABLES(a), else p_f

Let T = sum over all uncovered f of p_f * size_f (the same for every plan in a
round). Then v(a) = T - g(a) + 0.09 * sum_{f in ENABLES(a)} (1 - p_f) * size_f,
so

    Q(a) = (1 - gamma) * g(a) + gamma * 0.09 * B(a) + const,
    B(a) = sum_{f in ENABLES(a)} (1 - p_f) * size_f.

Consequences:
1. The "value of the next state" subtracts the immediate gain back out. At
   gamma=0.5 the ranking is g plus a bonus worth 9% of B; at gamma=1 the
   immediate term vanishes and only the bonus ranks plans (matches the drop at
   gamma=1 in the sweep: 82.6 vs 86.8-88.6).
2. The bonus is weighted by (1 - p_f): it prefers enabling functions we
   believe are UNLIKELY to be reachable.
3. One Beta posterior serves two meanings. A function predicted ENABLES and
   (correctly) not executed gets beta+1, lowering the p_f that the same
   formula says the plan raised. Both reviewers found this.
4. A covered function leaves the candidate set, so its alpha+1 is never read.
   Evidence from successes is thrown away.

So the flat gamma curve tests this implementation, not the idea of valuing
future reachability.

## Redesign: `cov_qvalue_tgt_fnt_bayes2` (NEW)

Keep generation, targeting and the EXECUTES/ENABLES scorer prompt unchanged.
Change only the valuation and the updates.

**Two quantities, each with one meaning.**

- Execution reliability r_f = P(f is covered after the plan | scorer said
  EXECUTES f). Per-function Beta, shrunk toward a global rate because one run
  has only 8 rounds:
      r_f = (a_f + m * rho) / (a_f + b_f + m),   m = 4
  rho = global hit rate of EXECUTES predictions in this run, itself
  Beta(1,1)-initialised. Updates use ONLY functions in EXECUTES of the
  committed plan: covered -> a_f+1 and rho hit; still uncovered -> b_f+1 and
  rho miss. Successes now matter through rho even though f leaves the
  candidate set (fixes point 4). ENABLES predictions never touch r_f (fixes
  point 3).
- Unlock lift delta = how much more likely an ENABLED function is to be covered
  in the next round than an uncovered function that was not enabled:
      u = P(covered in next round | predicted ENABLES by committed plan)
      b = P(covered in next round | uncovered, not predicted)
      delta = max(0, (u - b) / (1 - b))
  u and b are Beta posteriors with priors centred on the measured rates (0.212
  and 0.134, `analyze_future_conversion.py`) at prior strength 10, updated one
  round later when the outcome is known. This replaces the fixed UNLOCK_BUMP
  = 0.09 with a quantity learned per run, and gives the reviewers a direct
  answer to "what probability does the posterior represent".

**Q-value (no cancellation).**

    g(a) = sum_{f in EXECUTES(a)} r_f * size_f
    v(a) = h_t * delta * sum_{f in ENABLES(a)} r_f * size_f
    Q(a) = g(a) + gamma * v(a)

v is the expected coverage a follow-up round collects BECAUSE of this plan
(the increment over the base rate), weighted by r_f (prefers enabling
functions we can actually execute; fixes point 2). h_t = 1 if at least one
full round of budget remains after this plan, else 0: with no budget left
there is no future to value. This is the one-step form of
q = g + gamma * E[v(h')] with the plan-independent part of v(h') dropped, which
cannot change the argmax.

**Follow-through (`_ft`, exists).** Future value is only real if the next
round collects it. With `_ft`, next round's first target chunk is the enabled
functions plus the setup that ran. Test the redesign with and without it.

## Conditions

| ID | Strategy | What it tests |
|---|---|---|
| A0 | `cov_qvalue_tgt_rnd` | no selection (floor) |
| A1 | `cov_qvalue_tgt_fnt_bayes_g0` | immediate only, old posterior |
| A2 | `cov_qvalue_tgt_fnt_bayes` | current method |
| A3 | `cov_qvalue_tgt_fnt_bayes2_g0` (NEW) | immediate only, new r_f |
| A4 | `cov_qvalue_tgt_fnt_bayes2` (NEW) | redesign, gamma=0.5 |
| A5 | `cov_qvalue_tgt_fnt_bayes2_g1` (NEW) | redesign, gamma=1 |
| A6 | `cov_qvalue_tgt_fnt_bayes2_ft` (NEW) | redesign + follow-through |
| A7 | `cov_qvalue_tgt_fnt_bayes_oracle` | best-of-pool ceiling |

A3 vs A1: value of the cleaner posterior alone. A4/A5 vs A3: value of looking
ahead (Zakb Q1, 1zqK Q1). A6 vs A4: whether collecting the unlocked code
matters.

## Stages (cheapest first; all Gemma, $0)

**Stage 1, offline, no runs (2 h).** Re-score the logged candidate pools
(`*_calib` runs, `scripts/active/eval_scorers_offline.py`) with the new g.
Report top-1 selection accuracy and regret vs the old scorer (old: 54%
top-1). Also compute the *decision divergence*: the share of rounds where
argmax-Q at gamma=0.5 differs from argmax at gamma=0, for old and new
valuation. If divergence is a few percent, gamma cannot move coverage on this
benchmark no matter how v is defined; that number is itself an answer to the
reviewers. Go to stage 2 only if the new g is not worse offline.

**Stage 2, pilot (1.5 h).** A1-A6 on the 27-target stratified pilot set
(`--per-repo 3`), seed 42. Sanity checks: delta posterior moves, v > 0 in a
reasonable share of rounds (log it), no parsing regressions.

**Stage 3, hidden-key corridors (2 h).** Depths 4, 6, 8, seeds 42/123/456,
`--hidden`, with the `_rem` map (needed there, see `run_strategy` comment).
This is where a depth-d corridor needs d executions in order, so looking
ahead should matter most. Report stages reached and branches.

```bash
.venv/bin/python run_synth_corridor.py --hidden --depths 4 6 8 \
  --seeds 42 123 456 --exec-budget 24 \
  --strategies cov_qvalue_tgt_rem_fnt_bayes_g0 cov_qvalue_tgt_rem_fnt_bayes \
     cov_qvalue_tgt_rem_fnt_bayes2_g0 cov_qvalue_tgt_rem_fnt_bayes2 \
     cov_qvalue_tgt_rem_fnt_bayes2_ft \
  --output rebuttal_iclr27/E3_corridor_gemma.json
```

**Stage 4, RepoExploreBench (about 2.5 h per seed for 8 strategies).** All 93
targets, seeds 42/123/456, A0-A7 in one job so everything is paired.

```bash
.venv/bin/python run_repo_explore_bench.py --seeds 42 123 456 --parallel 48 \
  --strategies cov_qvalue_tgt_rnd cov_qvalue_tgt_fnt_bayes_g0 \
     cov_qvalue_tgt_fnt_bayes cov_qvalue_tgt_fnt_bayes2_g0 \
     cov_qvalue_tgt_fnt_bayes2 cov_qvalue_tgt_fnt_bayes2_g1 \
     cov_qvalue_tgt_fnt_bayes2_ft cov_qvalue_tgt_fnt_bayes_oracle \
  --output rebuttal_iclr27/E3_valuation_gemma.json
```

## Extra analyses from the same logs

- Calibration of r_f: predicted vs realised coverage of EXECUTES functions,
  in bins (reliability plot). Answers "what does the posterior represent".
- Trajectory of delta over rounds, per repo: where is unlocking real?
- Selection headroom captured: (ours - A0) / (A7 - A0); paper value is 17%.
- Per-target split by corridor structure (repos with deep setup chains vs
  flat modules), since the benefit should concentrate there.

## Decision rule (fixed before the run)

- **Adopt bayes2 (+ft if A6 >= A4)** as the method if A4 or A6 beats A3 on the
  corridors (p<0.05) and is not worse than A2 on RepoExploreBench (difference
  within noise or better). Then rerun the main table with it.
- **Adopt bayes2 at gamma=0 and reframe** if looking ahead shows no benefit
  even on corridors: contribution = posterior-targeted planning +
  function-grounded selection; future value reported as a negative result with
  the diagnosis above and the decision-divergence number.
- Either way the paper gets the two-posterior formulation, since it removes
  the inconsistency the reviewers identified at no cost.

## Honest prior

Selection captures 17% of oracle headroom and seed noise is about 3 branches,
so on RepoExploreBench the realistic outcome is "not worse, cleaner". The
corridors are where a real effect can show. Do not tune m, the prior strength,
or gamma on the 93 targets; fix them here (m=4, strength 10, gamma in
{0, 0.5, 1}) and report all.
