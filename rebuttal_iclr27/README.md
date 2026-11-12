# ICLR 2027 rebuttal experiments (Submission 48702)

Experiment specs for the ICLR 2027 rebuttal (reviewers Zakb, jDfu, 1zqK).

**This is a different rebuttal from the earlier one.** The earlier round's
material is `scripts/active/exp1..exp11`, `results/repo_explore_bench/exp*.log`
and the git-ignored `results/rebuttal/`; it concerned the previous version of
the method. Nothing here reuses those experiment numbers: this round uses
E1-E7, and its outputs go to `results/rebuttal_iclr27/`.
 One file per experiment; each lists the
reviewer question it answers, conditions, scale, commands, the analysis, and a
decision rule written down BEFORE the run. We report every result whichever
way it goes.

Code base: branch `fix-posterior-update` (frozen method `covqvalue2` =
`cov_qvalue_tgt_fnt_bayes`). Strategy names marked NEW do not exist yet.

| ID | Experiment | Answers | Paid LLM cost | Status |
|----|-----------|---------|---------------|--------|
| [E1](E1_same_information_baselines.md) | Baselines with the same target snippets | Zakb W3/Q2 | ~$40 (Gemini) | to implement (small) |
| [E2](E2_coverup_matched_budget.md) | CoverUp at a matched budget | 1zqK W4/Q4 | ~$20 (Gemini) | to implement |
| [E3](E3_future_value_redesign.md) | Lookahead headroom + redesigned future value and posteriors | Zakb W1/W2/Q1, 1zqK W1/W2/Q1/Q2 | $0 (Gemma) | to implement |
| [E4](E4_function_level_analysis.md) | Function coverage + gain decomposition | jDfu W3/Q3 | $0 (replay) | to implement |
| [E5](E5_branch_aware_targeting.md) | Branch-aware targeting (touched / rem / hybrid) | jDfu W1/Q1, 1zqK W3/Q3 | $0 (Gemma) | to implement |
| [E6](E6_mutation_analysis.md) | Fault detection by mutation analysis | 1zqK Q5 | $0 (Gemma suites) | to implement |
| [E7](E7_external_baseline_test4py.md) | New external baseline: Test4Py | jDfu W2/Q2 | $0 (Gemma) | setup, 1-day timebox |

## Budget policy

Paid API budget is tight, so the default is:

- **Gemma (local vLLM) carries every experiment**, with seeds 42/123/456. Free.
- **Gemini 3.8 Flash, one seed**, only for the two fairness experiments a
  reviewer will check across models (E1, E2). About $60 in total.
- **GLM is optional** (about $60 for E1, about $45 for E2). Run only if budget
  remains after E1/E2 on Gemini.
- No new TestGenEval runs (5 h and $50-90 per model).
- The final main-table rerun (E8, not specified here) waits for the E3 and E5
  decisions, so we pay for it once.

## Shared settings

Unless a spec says otherwise: RepoExploreBench, 93 targets, exec budget 24,
K=3, S=3, code `--gamma 0.5` (this is the paper's gamma = 1; see the mapping in E3), `DOCKER_MODE=exec`, `DOCKER_MEMORY=2g DOCKER_CPUS=1`.

```bash
cd /share/edc/home/aamayuelasfernandez/qcurious-tester
export MODEL=gemma-4-31B-it VLLM_API_BASE=http://localhost:8001/v1
export DOCKER_MODE=exec DOCKER_MEMORY=2g DOCKER_CPUS=1
```

Outputs go to `results/rebuttal_iclr27/<ID>_*.json` (+ `.log`).

Statistics: paired over targets (same test as the paper's ablation table),
seeds averaged per target first. Report mean, paired difference, 95% bootstrap
CI and p. Seed-to-seed noise on Gemma is about 3 branches on the 93-target
mean, so differences under ~3 branches are reported as "no detectable
difference", never as wins.

## Order

1. Tonight: E1 on Gemma (then Gemini), E4 replay.
2. Days 1-2: implement E3 and E5 (both Gemma, free), E2 hook, E6 pipeline,
   E7 setup (timeboxed).
3. After E3/E5 decisions: freeze the method, rerun the main table once.

## Known consistency issue (fix in the rerun)

`final_reb_*` (main table) started on Sep 22, before commit `52b691e`
("Use one covered rule for the Bayesian posterior update", Sep 23). The
ablations ran after it. E1 reruns `covqvalue2` on current code in the same job,
which gives paired numbers on one code version.

## Paper facts the specs rely on (submission PDF)

- Eq. 3 / Alg. 1: g = sum_E p_f|f|, v = lambda * sum_N (1-p_f)|f|, lambda =
  0.09, method gamma = 1; update over E union N (line 22); U = functions with
  no executed line (line 4).
- Fig. 5c: gamma 0 / 1/3 / 1 / v-only = 88.6 / 86.8 / 85.5 / 82.6; ladder
  35.4 -> 43.1 -> 80.0 -> 86.5 -> 85.5.
- Appendix E: EXECUTES accuracy 0.80, precision 0.49; ENABLES conversion
  17.8% vs 15.2%. This implies lambda of about 0.03, not 0.09 (E3 fixes this).
- Appendix F: 72 / 96 / 72 / 48 LLM calls (Random / Greedy / CovGreedy /
  ours), measured on the 10 click modules only. E1 extends it to all 93.
- CoverUp is "run to completion"; Pynguin gets 60 s (Table 3). E2 matches.
- Headline numbers are vs Greedy (no coverage feedback); 1zqK W4 objects.
  After E1/E2 the headline comparison should be vs the strongest baseline.
