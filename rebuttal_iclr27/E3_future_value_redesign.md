# E3. Future value: headroom, redesign, and posterior semantics

**Answers:** Zakb W1, W2, Q1; 1zqK W1, W2, Q1, Q2; jDfu's summary remark that
most of the gain comes from targeting.

## Notation (paper vs code)

The paper's Eq. 3 is  Q = g + gamma * v,  g = sum_{E} p_f |f|,
v = lambda * sum_{N} (1 - p_f) |f|, lambda = 0.09, default gamma = 1.
The code (`_rescore_with_posterior`) computes g + gamma_c * (value of the
whole next state), which reduces to (1 - gamma_c) * g + gamma_c * v + const.
Same ranking, with  gamma_paper = gamma_c / (1 - gamma_c):

| code flag | gamma_c | paper |
|---|---|---|
| `_g0` | 0 | gamma = 0 (88.6) |
| `_g025` | 0.25 | gamma = 1/3 (86.8) |
| default / `_g05` | 0.5 | gamma = 1, the method (85.5) |
| `_g1` | 1 | "v only, no g" (82.6) |

New variants below take the PAPER gamma directly.

## What the paper already concedes, and what is actually wrong

Section 5.1 already says the future-value term "has no measurable effect" and
Appendix E reports that ENABLES functions are covered next round 17.8% of the
time vs 15.2% for other untested functions. The reviewers' objection is that
the title, abstract and corridor motivation rest on a term with no measured
benefit, and that the posterior is not well defined. Four concrete problems:

1. **Weak signal, small weight.** v is an ENABLES bonus scaled by 0.09, so it
   rarely changes the argmax; and the ENABLES predictions themselves carry a
   2.6-point lift. Nobody has measured whether a better signal COULD help on
   this benchmark (no headroom estimate).
2. **lambda is inconsistent with the paper's own numbers.** lambda = 0.09
   comes from pilot logs (21.2% vs 13.4%). The rates in Appendix E give
   (0.178 - 0.152) / (1 - 0.152) = 0.03. A reviewer can compute this. Fix by
   learning the lift per run instead of fixing a constant.
3. **One posterior, two meanings** (Zakb W2, 1zqK W2). Algorithm 1 line 22
   updates over E union N: a function predicted ENABLED and, as predicted, not
   executed gets beta+1, lowering the p_f that Eq. 3 says the plan raised.
4. **Successes are discarded** (Zakb W2). A covered function leaves U, so its
   alpha+1 is never read; p_f only ever moves down.

Also: the (1 - p_f) weight in v favours enabling functions we believe are
hard to reach, with no check that they are then executable.

## Stage 0. Is there any headroom for looking ahead? (new, most important)

Before redesigning v, measure the ceiling. Two counterfactual selectors on the
SAME plan pool, trial runs rolled back from a snapshot (free lookahead, as in
the existing `_oracle`):

- `..._oracle` (exists): commit the plan with the highest realised gain this
  round. Myopic ceiling.
- `..._oracle2` (NEW): for each of the K plans, run it, then from the
  resulting state generate the next round's K plans and trial-run them;
  commit the plan maximising (gain now + best gain next round). Two-round
  lookahead ceiling.

Headroom of planning = oracle2 - oracle. Run on the 27-target pilot set
(seed 42; LLM calls are about 4x, free on Gemma, about 3 h) and on the
hidden-key corridors (depths 4/6/8).

- If oracle2 is about equal to oracle on RepoExploreBench: with K=3 targeted
  plans, no future-value estimator can help there; the benefit of planning is
  in the multi-step PLANS (S=3) and in targeting, not in the selector. That is
  a clean, honest answer to Zakb Q1 / 1zqK Q1, and the corridor result shows
  where lookahead does matter.
- If oracle2 is clearly above oracle: there is something to capture, and the
  redesign below is tested against that ceiling.

## Redesign: `cov_qvalue_tgt_fnt_bayes2` (NEW)

Generation, targeting and the EXECUTES/ENABLES prompt are unchanged.

**Execution reliability r_f** = P(f covered after the plan | predicted
EXECUTES f). Per-function Beta shrunk to a run-level rate, since a run has 8
rounds:

    r_f = (a_f + m * rho) / (a_f + b_f + m),   m = 4

rho = hit rate of EXECUTES predictions so far in this run (Beta(1,1) start;
Appendix E precision 0.49 says it should settle near 0.5). Updates use only
functions in E of the committed plan. Hits raise rho, so evidence from
successes is used even though f leaves U (problem 4). ENABLES predictions
never touch r_f (problem 3). This is the answer to "what probability does the
posterior represent": the probability that a function the scorer says a plan
executes is in fact covered.

**Unlock lift delta**, learned per run (problem 2):

    u = P(covered next round | predicted ENABLES by the committed plan)
    b = P(covered next round | untested, not predicted)
    delta = max(0, (u - b) / (1 - b))

u and b are Beta posteriors with priors centred on the Appendix E rates
(0.178, 0.152), prior strength 10, updated one round later.

**Q-value.**

    g(a) = sum_{f in E(a)} r_f |f|
    v(a) = h_t * delta * sum_{f in N(a)} r_f |f|
    Q(a) = g(a) + gamma * v(a),   gamma = 1

v is the expected extra code the next round covers because of this plan,
weighted by how reliably we execute what we target (replaces the 1 - p_f
weight). h_t = 0 in the last round (no future to value), else 1.

**Follow-through (`_ft`, exists).** Next round's first chunk is the functions
the committed plan enabled, with the setup that ran. Future value is only
real if the next round goes and collects it.

## Conditions

| ID | Strategy | Tests |
|---|---|---|
| A0 | `cov_qvalue_tgt_rnd` | no selection |
| A1 | `cov_qvalue_tgt_fnt_bayes_g0` | paper, gamma = 0 |
| A2 | `cov_qvalue_tgt_fnt_bayes` | paper method (gamma = 1) |
| A3 | `cov_qvalue_tgt_fnt_bayes2_g0` (NEW) | new r_f, immediate only |
| A4 | `cov_qvalue_tgt_fnt_bayes2` (NEW) | redesign |
| A5 | `cov_qvalue_tgt_fnt_bayes2_ft` (NEW) | redesign + follow-through |
| A6 | `cov_qvalue_tgt_fnt_bayes_oracle` | myopic ceiling |

A3 vs A1: the cleaner posterior alone. A4, A5 vs A3: looking ahead.

## Stages after Stage 0 (all Gemma, $0)

1. **Offline (2 h, no runs).** Re-score logged pools
   (`eval_scorers_offline.py`) with the new g: top-1 accuracy and regret vs
   the current scorer. Compute decision divergence: share of rounds where the
   argmax at gamma = 1 differs from gamma = 0, old and new valuation.
2. **Pilot.** A1-A5, 27 targets, seed 42. Check delta moves, share of rounds
   with v > 0, no parsing regressions.
3. **Hidden-key corridors.** Depths 4/6/8, seeds 42/123/456, `--hidden`, with
   the `_rem` map (required there). Stages reached and branches.
4. **RepoExploreBench.** 93 targets, seeds 42/123/456, A0-A6 in one job.

```bash
.venv/bin/python run_repo_explore_bench.py --seeds 42 123 456 --parallel 48 \
  --strategies cov_qvalue_tgt_rnd cov_qvalue_tgt_fnt_bayes_g0 \
     cov_qvalue_tgt_fnt_bayes cov_qvalue_tgt_fnt_bayes2_g0 \
     cov_qvalue_tgt_fnt_bayes2 cov_qvalue_tgt_fnt_bayes2_ft \
     cov_qvalue_tgt_fnt_bayes_oracle \
  --output rebuttal_iclr27/E3_valuation_gemma.json
```

## Extra analyses from the same logs

- Reliability plot for r_f (predicted vs realised), and delta over rounds per
  repo. Extends Appendix E.
- Selection headroom captured: (method - A0) / (A6 - A0).
- Split by repo: where is unlocking real?

## Decision rule (fixed before the runs)

- Adopt bayes2 (+ft if A5 >= A4) if it beats A3 on the corridors (p<0.05)
  and is not worse than A2 on RepoExploreBench. Rerun the main table with it.
- If looking ahead shows nothing even on corridors, or Stage 0 shows no
  headroom: keep bayes2 at gamma = 0 for its well-defined posterior, and
  reframe: the planning that matters is multi-step targeted plans chosen from
  the exploration state; future-value selection is a reported negative result
  with a measured ceiling. Title/abstract wording adjusted accordingly.
- Either way the revised paper uses the two-quantity formulation and the
  learned lift, which remove problems 2-4 at no cost.

Fixed in advance, not tuned on the 93 targets: m = 4, prior strength 10,
gamma in {0, 1}.

## Honest prior

A 2.6-point ENABLES lift and seed noise of about 3 branches make a measurable
win on RepoExploreBench unlikely. The realistic strong outcome is: Stage 0
quantifies the headroom, the redesign is at least as good with a posterior
that means something, and the corridors show a real effect of lookahead.
