# Calibration Results — Phases 1 through 0b

**Date range:** 2026-03-22 to 2026-03-23
**Benchmark:** ULT (30–97 functions, cyclomatic complexity 20+)
**Total cost:** ~$56 across all calibration experiments

---

## 1. What We Were Testing

We wanted to know: can we build an estimator that scores K candidate tests so we can pick the best one? We tried multiple estimators and multiple evaluation targets.

### Estimators tested

| Estimator | Method | Models | Cost per candidate |
|---|---|---|---|
| Single-model sampling entropy | S predictions at temp=0.9, Shannon entropy | Gemini, Mistral | S API calls |
| Coverage-prediction disagreement | S predictions of "will this hit new branches?", binary entropy | Gemini, Mistral | S API calls |
| Multi-model disagreement | 1 prediction per model, entropy across models | Gemini + Mistral + GPT-5.4-mini | 3 calls |
| Token-level logprob entropy | 1 prediction with logprobs, mean per-token entropy | gpt-oss-120b (Fireworks) | 1 call |
| Logprob entropy + z-score | Same as above, z-scored within each batch of K candidates | gpt-oss-120b | 1 call |
| Verbalized confidence | Ask model to rate confidence 0-100 | Gemini | 1 call |
| Contrastive ranking | Show all K candidates, ask model to rank them | Gemini | 1 call |
| P(yes) from logprobs | Ask "will this find new branches?", extract P(yes) from logprobs | gpt-oss-120b | 1 call |

### Evaluation targets

| Target | What it measures |
|---|---|
| Immediate coverage gain (`new_branches`) | How many new branches does this test cover right now? |
| Prediction surprise | Was the model's predicted output wrong? (learning signal) |
| Next-step quality | After incorporating this test, are the next candidates better? |

### Evaluation metric

**Within function×state Spearman ρ**: among K candidates for the same function at the same exploration state, does the estimator rank them in the same order as the target? This is the decision our algorithm actually needs to make.

---

## 2. Phase 1 Results — Single-Model Sampling Entropy

**Config:** 97 functions, K=10, S=6 then S=20, states [0, 5, 10]
**Target:** Immediate coverage gain

| Model | Overall ρ | 20+ Complexity ρ | Cost |
|---|---|---|---|
| Gemini 3 Flash (S=6) | +0.081 | +0.148 | $12.69 |
| Mistral Large (S=6) | -0.089 | -0.053 | $8.72 |
| Gemini 3 Flash (S=20) | — | +0.054 | $24.72 |
| Mistral Large (S=20) | — | +0.095 | $4.39 |

**Verdict: PIVOT.** Increasing S from 6 to 20 made Gemini worse (0.148 → 0.054), suggesting S=6 result was noise. Fundamental problem: 71% of candidates produce zero entropy (model gives identical predictions across all samples).

**Coverage-prediction disagreement** was also tested (S=20, 20+ complexity): Gemini ρ=-0.289, Mistral ρ=-0.106. **STOP.** Models unanimously predict "yes, new branches" for every candidate.

---

## 3. Phase 0b v1 Results — Alternative Estimators

**Config:** 5 functions (smoke test), K=5, states [0, 5, 10]
**Target:** Immediate coverage gain
**New estimators:** Multi-model disagreement, logprob entropy, verbalized confidence

| Estimator | Overall ρ | Within func×state ρ | Verdict |
|---|---|---|---|
| Multi-model disagreement | +0.12 | ~0 (formatting noise) | STOP |
| Logprob entropy (raw) | -0.28 | +0.36 (3/4 positive) | Misleading |
| Verbalized confidence | +0.34 | -0.46 (0/2 positive) | Misleading |

### Key finding: overall ρ was misleading

**Verbalized confidence** appeared to be the winner (ρ=+0.34, p=0.005). But trace analysis revealed:
- Output confidence was always 100 (model always claims certainty)
- Branch confidence was binary: 0 or 100
- Within function×state, all candidates received the **same score** — no candidate discrimination
- The positive overall ρ was entirely driven by cross-state confounding (state 0 = high gain + high score, state 10 = low gain + low score)

**Logprob entropy** appeared to fail (ρ=-0.28). But:
- The anti-correlation was cross-function confounding (complex-output functions have high baseline entropy AND less coverage headroom)
- Within function×state, 3/4 groups showed positive correlation (mean ρ=+0.36)
- The real signal was hidden by the confound

**Multi-model disagreement** failed because:
- Gemini wraps string outputs in quotes, GPT/Mistral don't → spurious formatting disagreement
- Only 3 models → only 3 possible entropy values (0, 0.92, 1.58) — too coarse
- After normalizing predictions (stripping quotes): still no improvement

This analysis led us to adopt **within function×state ρ** as the primary metric instead of overall ρ.

---

## 4. Phase 0b v2 Results — Improved Estimators at Scale

**Config:** 30 functions, K=10, states [0, 5, 10]
**Target:** Immediate coverage gain

| Estimator | Within func×state ρ | Positive groups | Verdict |
|---|---|---|---|
| Logprob entropy + z-score | +0.047 | 24/47 (51%) | STOP |
| Contrastive ranking | +0.102 | 28/47 (60%) | PIVOT |
| P(yes) from logprobs | -0.139 | 14/38 (37%) | STOP |

**Cost:** $1.87, 33 minutes

### Why P(yes) failed

gpt-oss-120b is a reasoning model. It chains through ~150 tokens of thinking before answering yes/no. The final answer token has P(yes)≈1.0 or P(no)≈1.0 (deterministic after reasoning). We attempted extracting P(yes) from the first reasoning token where yes/no appeared, but this captured the model's surface-level assessment, not deep understanding of branch reachability.

### Why the smoke test numbers didn't hold

| Estimator | 5-func run 1 | 5-func run 2 | 30-func run |
|---|---|---|---|
| Logprob entropy + z-score | +0.80 (4 groups) | +0.18 (3 groups) | +0.05 (47 groups) |
| Contrastive ranking | +0.48 (4 groups) | +0.09 (3 groups) | +0.10 (47 groups) |

With 3-4 groups, a single group flipping sign changes the mean by 0.5+. The 30-function run is the reliable number. **Lesson: never trust smoke tests with < 10 usable groups.**

---

## 5. Phase 0b v3 Results — Different Evaluation Target

**Insight:** We had been evaluating whether the estimator predicts immediate coverage gain. But information gain in Sun et al.'s framework is prediction surprise — I(O; Θ|h,a). The most informative test is the one whose outcome the model is most wrong about, not the one that covers the most branches.

**Config:** 30 functions, K=10, states [0, 5, 10]
**Estimator:** Logprob entropy + z-score

| Target | Within func×state ρ | Positive groups | Verdict |
|---|---|---|---|
| Immediate coverage gain | -0.15 | 19/46 (41%) | STOP |
| Prediction surprise | -0.11 | 30/66 (45%) | STOP |
| Next-step quality (high-ent vs low-ent pick) | 14 wins / 18 losses / 55 ties | 16% win rate | STOP |

**Cost:** $2.29, 28 minutes

**Prediction surprise also did not hold at scale.** The smoke test showed ρ=+0.32 (6/9 positive) but the full run showed ρ=-0.11 (30/66 positive ≈ coin flip). Same pattern as before: small samples are noise.

**Next-step quality** was mostly ties (55/87 = 63%) because coverage saturated — neither the high-entropy nor low-entropy pick led to any further progress. Among non-tie cases, low-entropy picks won slightly more often (18 vs 14).

**Prediction accuracy** was 1% (11/866 exact matches). The model is almost always wrong about the output, but being wrong about one candidate vs another doesn't predict anything useful.

---

## 6. Fundamental Diagnosis

### What we learned about the estimators

**No single-step estimator reliably discriminates candidates within a function.** Across 7 estimators and 3 evaluation targets, the best within-group ρ at scale was +0.10 (contrastive ranking vs immediate gain). Token-level logprob entropy contains real variance but does not track any target we measured.

### Why the calibration approach is limited

The calibration tests whether an estimator predicts **single-step outcomes**: which candidate will cover more branches right now, or which will surprise the model right now. But the theory (Sun et al.) argues information gain is valuable because of its **cumulative multi-step effect**:

```python
def process_order(order):
    # CORRIDOR: validation gates
    if not isinstance(order, dict): return {"error": "not_a_dict"}
    if "items" not in order:       return {"error": "missing_fields"}

    # DEEP LOGIC: only reachable after passing validation
    total = sum(item["price"] * item["qty"] for item in order["items"])
    if total > 1000: ...
```

A test that passes validation but hits zero new branches (immediate gain = 0, surprise = low because the model predicted the correct shallow output) can still be the most valuable test — it teaches the model how to construct valid inputs, enabling much better candidates in the next round.

**This corridor effect is invisible to single-step calibration.** It can only be measured by running the full exploration loop over multiple steps and comparing cumulative coverage curves.

### What the calibration DID establish

1. **Single-model sampling entropy fails** — 71% zero-entropy floor, model-specific, inverts at later states
2. **Verbalized confidence is a state detector, not a candidate discriminator** — binary scores, no within-group variance
3. **Multi-model disagreement needs semantic comparison** — formatting differences create spurious disagreement
4. **Reasoning models are poor for P(yes)** — they commit during reasoning, final token is deterministic
5. **Small samples are unreliable** — 5-function smoke tests produced ρ values ranging from -0.3 to +0.8 for the same estimator
6. **Overall ρ masks confounds** — cross-function and cross-state effects dominate within-group signal

---

## 7. Next Step: Head-to-Head Strategy Comparison

Instead of more calibration, run the full exploration loop with competing strategies on the same functions and compare cumulative coverage:

| Strategy | Selection rule |
|---|---|
| **Random** | Pick uniformly from K candidates |
| **Greedy coverage** | Ask LLM which candidate will cover the most new branches, pick its top choice |
| **Curiosity (logprob entropy)** | Pick the candidate with highest token-level logprob entropy |
| **Oracle** | Execute all K candidates, pick the one with highest actual coverage gain |

**Protocol:** For each of N functions, run each strategy for B=15 steps (budget). At each step: generate K=5 candidates, select one by the strategy's rule, execute it, update history. Compare cumulative branch coverage at steps 1, 5, 10, 15.

This directly tests whether curiosity-guided selection leads to better exploration than greedy — including the corridor effect that calibration cannot measure.

---

## 8. Cost Summary

| Experiment | Cost | API Calls |
|---|---|---|
| Phase 1: Single-model entropy (Gemini, S=6) | $12.69 | 22,928 |
| Phase 1: Single-model entropy (Mistral, S=6) | $8.72 | 18,371 |
| Phase 1: Both estimators (Gemini, S=20) | $24.72 | — |
| Phase 1: Both estimators (Mistral, S=20) | $4.39 | — |
| Phase 0b v1: Alternative estimators (smoke tests) | $0.60 | ~1,000 |
| Phase 0b v2: Improved estimators (smoke tests + full) | $2.24 | ~4,400 |
| Phase 0b v3: Two-target calibration (smoke + full) | $2.51 | ~3,950 |
| **Total calibration** | **~$56** | **~51,000** |
