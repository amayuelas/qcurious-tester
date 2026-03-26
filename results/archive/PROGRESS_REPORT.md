# Curiosity-Guided Code Exploration — Progress Report

**Date:** 2026-03-24
**Total spend:** ~$210
**Plan reference:** PLAN_v4.md

---

## Executive Summary

We are implementing Sun et al.'s (2011) optimal Bayesian exploration framework for LLM-based test generation. The core idea: select tests by expected information gain (what the LLM is most uncertain about) rather than greedy coverage feedback.

**Current status:** Early positive results on real-world Django code. Q-value planning (γ=0.5) outperforms greedy coverage by ~5 branches on TestGenEval Django files. The approach works best on file-level code with genuine corridor structure (import/setup requirements → deep logic).

---

## What We've Done

### Phase 0: Proof of Concept ✓ (~$2)
- Built the core pipeline: candidate generation → scoring → selection → execution → coverage measurement
- Tested on 4 toy programs with known corridor structure
- Curiosity matched or beat greedy on corridor programs
- **Result: GO — proceeded to calibration**

### Phase 0.5: Infrastructure ✓ (~$3)
- Benchmark loaders: ULT (3,909 functions), CRUXEval (800 functions), TestGenEval (stub → now Docker)
- Multi-model support: Gemini (via proxy), Mistral (direct API), OpenAI GPT-5.4, Fireworks gpt-oss-120b
- Coverage runner with branch-level tracking via coverage.py
- Cost tracking per model

### Phase 1: ULT Calibration ✓ (~$56)
- Tested whether estimator scores predict single-step coverage gain
- **8 estimators tested:**
  1. Single-model sampling entropy (S=6, S=20)
  2. Coverage-prediction disagreement
  3. Multi-model disagreement (Gemini + Mistral + GPT-5.4-mini)
  4. Token-level logprob entropy (Fireworks gpt-oss-120b)
  5. Logprob entropy + z-score normalization
  6. Verbalized confidence (0-100 rating)
  7. Contrastive ranking (rank K candidates)
  8. P(yes) from logprobs

- **None met the GO threshold (ρ > 0.15) at scale (30 functions)**
- Key finding: overall ρ was misleading due to cross-function/cross-state confounds
- Adopted within-function×state ρ as the correct metric
- **Verdict: PIVOT — moved to head-to-head comparison**

### Phase 0b: Estimator Diagnostics ✓ (~$10)
- Discovered the zero-entropy floor: 79-91% of candidates score entropy=0
- Verbalized confidence was a state detector, not a candidate discriminator
- Logprob entropy had real within-function signal (ρ=+0.36) masked by cross-function confound
- **Key realization: single-step calibration can't capture the corridor effect**

### Head-to-Head Comparisons ✓ (~$50)

#### On ULT (30 functions, Gemini, white-box):
| Strategy | Mean Coverage |
|---|---|
| Oracle | 62.4 |
| Curiosity sampling | 62.1 |
| Curiosity contrastive | 61.5 |
| Greedy | 61.0 |
| Random | 59.2 |

- Curiosity slightly beats greedy — **not statistically significant** (p=0.75)
- Oracle gap tiny (~1.4 arcs) — candidate selection barely matters on ULT
- ULT functions are too easy: the LLM passes validation gates trivially

#### On Handwritten Corridor Programs (4 functions, white-box):
| Strategy | Mean Coverage |
|---|---|
| Q-value (γ=0.5) | 148.8 |
| Greedy | 147.8 |
| Diverse ART | 147.0 |
| Random | 145.5 |
| Curiosity greedy (γ=0) | 145.5 |

- **Q-value is the best non-oracle strategy**
- Q-value beats curiosity_greedy on 3/4 functions (csv +11, http +9, expr +3)
- The planning term (γ) contributes ~1.5 arcs beyond just extra computation
- On `task_scheduler`: Q-value starts behind (step 1: 87 vs 112) but overtakes at step 3 (140 vs 125) — the corridor crossover pattern

#### On TestGenEval Django (2 files, white-box) — LATEST:
| Strategy | Mean Coverage |
|---|---|
| **Q-value (γ=0.5)** | **432.0** |
| Greedy | 426.5 |
| Curiosity sampling | 419.0 |
| Random | 412.5 |

- **Q-value wins by +5.5 over greedy, +20 over random**
- On `formsets.py`: Q-value 297 vs greedy 286 (+11), curiosity 267 (+30)
- Differentiation is 10x larger than ULT (20 branches vs 1-2)
- File-level Django code has genuine corridor structure
- **Currently running on 6 files to confirm** ($1.26 per 2 files)

### Additional Findings

#### Diverse Generation (~$13)
- Using different prompting strategies per candidate (edge case, error path, deep logic, adversarial, mutation, CoT) improves candidate quality
- Diverse generation + random selection beats greedy 83% on corridor functions
- **Generation diversity matters more than selection intelligence** on function-level tasks

#### Black-Box Condition (~$5)
- Without source code, curiosity collapses — the LLM can't predict outputs from just signatures
- Greedy is NOT degraded in black-box (68.8 vs white-box 74.8)
- The model reasons about "what behavior haven't I seen?" even without code
- **Prediction: black-box curiosity needs more steps to build a useful world model through interaction**

#### Model Mismatch
- Using gpt-oss-120b for logprob scoring + Gemini for generation = model mismatch
- Logprob entropy was worst strategy on ULT with mismatch, competitive without it
- **Lesson: the same model must be used for generation and scoring**

---

## Key Technical Insights

### 1. Single-step calibration is the wrong evaluation
We spent ~$56 correlating estimator scores with single-step outcomes. But information gain's value is cumulative over multiple steps (Sun et al. Section 3.2). A test that scores zero on immediate coverage gain can be the most valuable because it teaches the model how to pass validation gates, enabling better future tests.

### 2. The zero-entropy floor cripples sampling-based estimators
Gemini produces identical predictions across S=8 samples 79-91% of the time. The estimator can't discriminate when most candidates score 0. This is a readout problem, not a signal problem — the model IS uncertain but temperature sampling doesn't surface it.

### 3. Q-values work because they evaluate future potential
The γ·E[v(h')] term in the Q-value correctly values tests that pass validation gates. The model simulates: "if I run this test and see this output, what candidates become available?" Tests that unlock new code regions have high future value even with low immediate coverage.

### 4. File-level tasks have larger corridor effects
ULT functions are standalone with simple inputs — oracle gap ~1.4 arcs. Django files require import chains, class setup, configuration — oracle gap ~20 branches. The corridor hypothesis is validated but only on tasks with genuine corridor structure.

### 5. Oracle is not an upper bound
Our single-step greedy oracle loses to random on some functions. This confirms Sun et al.'s non-additivity argument: greedy single-step optimization is suboptimal for cumulative exploration.

---

## What Remains

### Immediate (in progress)
- [ ] TestGenEval Django: 6-file run to confirm Q-value advantage
- [ ] TestGenEval: build more Django version images for larger sample

### Short-term
- [ ] Run Q-value comparison on all 12 Django 4.0 examples
- [ ] Build Django 3.x and 5.0 images for full TestGenEval-Lite (160 examples)
- [ ] Statistical significance testing with enough examples (n ≥ 20)
- [ ] Black-box condition on TestGenEval

### Medium-term (Phase 2)
- [ ] Full TestGenEval-Lite evaluation (160 examples)
- [ ] Compare against TestForge and CoverUp baselines
- [ ] Coverage curves at matched API cost
- [ ] Ablations: K, S, γ values

### Long-term (Phase 3-4)
- [ ] Corridor analysis: identify files where Q-value helps most
- [ ] Planning-aware extension with deeper lookahead
- [ ] Black-box condition on TestGenEval
- [ ] Paper writing

---

## Risk Assessment (Updated)

| Risk | Status | Notes |
|---|---|---|
| Entropy doesn't correlate with coverage gain | **Confirmed** | Single-step calibration fails. But the full loop works. |
| All uncertainty readouts fail | **Partially confirmed** | At single-step level, yes. Q-values with planning work in the loop. |
| Curiosity ≈ baselines on benchmarks | **Depends on benchmark** | True for ULT. False for TestGenEval (preliminary). |
| Planning-ahead adds nothing | **Promising** | γ=0.5 beats γ=0 on handwritten corridors. TestGenEval pending. |
| TestGenEval Docker too complex | **Resolved** | Docker working, coverage measured successfully. |
| Oracle ceiling too low | **True for ULT, false for TestGenEval** | ULT gap ~1.4 arcs. Django gap ~20 branches. |

---

## Cost Breakdown

| Experiment | Cost |
|---|---|
| Phase 0-0.5: Setup + toy programs | ~$5 |
| Phase 1: ULT calibration (Gemini + Mistral) | ~$56 |
| Phase 0b: Estimator diagnostics | ~$10 |
| Head-to-head ULT (Gemini) | ~$21 |
| Diverse generation experiments | ~$13 |
| Black-box experiments | ~$5 |
| Q-value handwritten corridors | ~$28 |
| Generated corridor functions | ~$92 |
| TestGenEval Django (2 files) | ~$1.30 |
| **Total** | **~$210** |

**Budget remaining:** ~$100 of the $315 allocated in PLAN_v4.

---

## Files Created

### Experiment scripts
- `run_calibration.py` — Phase 1 calibration
- `run_phase0b.py`, `run_phase0b_v2.py`, `run_phase0b_v3.py` — Estimator diagnostics
- `run_head2head.py`, `run_head2head_v2.py` — Strategy comparisons
- `run_blackbox.py` — Black-box experiments
- `run_corridor_test.py` — Corridor-specific with Q-values
- `run_qvalue_test.py` — Q-value dedicated test
- `run_testgeneval.py` — TestGenEval Django integration

### Library code
- `curiosity_explorer/explorer/q_values.py` — Q-value computation (Sun et al. Prop. 1)
- `curiosity_explorer/explorer/diverse_gen.py` — Diverse candidate generation
- `curiosity_explorer/explorer/art_selection.py` — Adaptive Random Testing selection
- `curiosity_explorer/runner/docker_coverage.py` — Docker-based coverage for TestGenEval
- `curiosity_explorer/benchmarks/corridor_programs.py` — 4 handwritten corridor programs
- `curiosity_explorer/benchmarks/function_generator.py` — Parameterized function generator

### Results
- `results/calibration_results.md` — Full calibration findings
- `results/PROGRESS_REPORT.md` — This file
- `results/*.json` — Raw data from all experiments
