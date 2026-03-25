# Curiosity-Guided Code Exploration
## Optimal Bayesian Exploration for LLM-Based Test Generation

### Revised Research Plan v4 — March 2026

---

## 1. What We're Doing

We apply Sun et al.'s (2011) optimal Bayesian exploration framework to LLM-based test generation. Instead of selecting tests by greedy coverage feedback (what existing tools do), we select tests by **expected information gain** — choosing the test whose outcome the LLM is most uncertain about, because that test will reveal the most about the program's unknown behavior.

The key theoretical contribution is that information gain is **additive only in expectation** (Sun et al., Section 3.2), which means you can't just substitute it as a reward in standard MCTS. The practical contribution is that this enables **planning through coverage corridors** — sequences of inputs that individually yield zero new coverage (passing validation gates) but unlock access to deep, unexplored branches.

---

## 2. Why Code Exploration

We considered several settings (math reasoning, agent exploration in text games, API discovery, scientific hypothesis search) and chose code exploration because:

**Exploration IS the metric.** Coverage directly measures exploration. We don't have the problem where "curiosity explored more but accuracy didn't improve" — if coverage goes up, the method works.

**The environment is genuinely unknown.** The LLM doesn't know what a program does until it runs tests. This is a direct instantiation of Sun et al.'s unknown-MDP formulation, not an analogy. (In math reasoning, the LLM already knows algebra — the "dynamics" aren't unknown, only the answer is.)

**Corridors are real and measurable.** Programs with validation gates have literal corridor structure. We can identify corridors via cyclomatic complexity and nesting depth, and measure whether our method traverses them.

**Benchmarks and baselines already exist.** TestGenEval (Meta Research, 1,210 file pairs, GPT-4o at 35% coverage) and ULT (3,909 functions, curated for high complexity) provide unsaturated evaluation. CodaMosa, CoverUp, and TestForge provide strong baselines. We don't need to build evaluation infrastructure from scratch.

**No saturation problem.** Deep branches in complex functions are genuinely hard to reach. GPT-4o achieves only 35.2% average coverage on TestGenEval and <30% branch coverage on ULT's high-complexity functions — enormous headroom for improvement, unlike math benchmarks at 95%+.

---

## 3. Formal Mapping

| Sun et al. (2011) | Code Exploration |
|---|---|
| Unknown environment parameter Θ | Unknown program behavior (input-output mapping, branch conditions, edge cases) |
| Action a | Test input x |
| Observation o | Execution trace: output, exceptions, branches hit |
| History h = a₁o₁...aₜoₜ | Sequence of (test input, execution result) pairs |
| p(θ\|h) | LLM's beliefs about program behavior given tests so far |
| ḡ(a\|h) = I(O; Θ\|h,a) | Expected information about the program gained by running test x |
| Corridor (deterministic, zero info gain) | Validation gates (must pass, no new coverage) |
| Clique (stochastic, high info gain) | Deep branch logic (many behaviors to discover) |
| Cumulative information gain | Cumulative branch coverage |

This mapping is a **direct instantiation**, not an analogy. The program IS the unknown MDP. Sun et al.'s theory applies without modification.

**Architectural note (updated):** The LLM + test history in context serves as the world model M. The history IS the Bayesian update mechanism — in-context learning serves as the posterior update M → M* from Schmidhuber's 1995 formulation. The open problem is not "do we have a world model?" but "how do we read out M's uncertainty?" (see Section 5.1 and Section 6).

---

## 4. The Corridor Problem in Code

```python
def process_order(order):
    # CORRIDOR: validation gates (must pass, zero new coverage after first time)
    if not isinstance(order, dict):
        return {"error": "not_a_dict"}
    if "items" not in order or "customer_id" not in order:
        return {"error": "missing_fields"}
    if not isinstance(order["items"], list) or len(order["items"]) == 0:
        return {"error": "invalid_items"}

    # CLIQUE: deep business logic (many branches to discover)
    total = sum(item["price"] * item["quantity"] for item in order["items"])
    if total > 1000:
        if order["customer_id"] > 100:
            return {"status": "premium_large_order", "total": total}
        else:
            return {"status": "large_order", "review": True}
    elif total > 100:
        return {"status": "medium_order", "total": total}
    ...
```

**Greedy coverage** quickly covers validation error branches, then plateaus — every candidate that passes validation looks like "zero new branches" until deep logic is actually reached.

**Curiosity-guided selection** recognizes that a valid input has high information gain (the LLM is uncertain what the deep logic will produce), and invests in constructing such inputs even though immediate coverage gain is zero.

**Planning-aware curiosity** (curiosity Q-values) goes further: it values corridor-traversing inputs for their *future* information content, not just immediate uncertainty.

---

## 5. Algorithm

### 5.1 Information Gain Estimation

The core challenge: estimating ḡ(a|h) = I(O; Θ|h,a) — the mutual information between the observation and the unknown environment parameter, given history and a candidate action.

**The world model is the LLM + history.** The LLM's beliefs about program behavior update through in-context learning as test results accumulate in the prompt. This is a legitimate instantiation of Schmidhuber's M → M* update: the history h grows, and the LLM's predictions p(o|h,a) change accordingly.

**The open problem is the uncertainty readout.** We need to measure how uncertain M is about each candidate's outcome. The original estimator (output prediction entropy via temperature sampling) failed calibration on realistic code (see Section 6). We are evaluating four alternative readout methods:

#### Estimator A: Multi-Model Disagreement (primary candidate)

```python
def information_gain_ensemble(models, context, candidate_test):
    """Use disagreement across different LLMs as epistemic uncertainty proxy."""
    predictions = [
        model.predict_output(context, candidate_test)
        for model in models  # e.g., Gemini, Mistral, Qwen, Llama, DeepSeek
    ]
    # Cluster semantically equivalent predictions
    clusters = cluster_predictions(predictions)
    # Compute entropy across model predictions
    probs = [len(c) / len(models) for c in clusters]
    return -sum(p * log(p) for p in probs if p > 0)
```

**Why this should work:** Different models have different inductive biases, training data, and architectures. Their disagreement approximates *epistemic* uncertainty (genuine uncertainty about program behavior), unlike single-model temperature sampling which captures *aleatoric* variation (token-level randomness). This directly addresses the zero-entropy problem — even if Gemini is deterministic across 20 samples, Gemini and Mistral may predict different outputs.

#### Estimator B: Token-Level Logprob Entropy

```python
def information_gain_logprobs(model, context, candidate_test):
    """Use token-level logprobs from a single prediction as continuous uncertainty."""
    # Generate one output prediction with logprobs enabled
    response = model.predict_output(
        context, candidate_test, return_logprobs=True
    )
    # Extract per-token entropy from logprob distribution
    token_entropies = []
    for token_logprobs in response.logprobs:
        # token_logprobs: dict of {token: log_prob} for top-k candidates
        probs = [exp(lp) for lp in token_logprobs.values()]
        token_entropy = -sum(p * log(p) for p in probs if p > 0)
        token_entropies.append(token_entropy)
    # Aggregate: mean entropy across output tokens
    return sum(token_entropies) / len(token_entropies)
```

**Why this should work:** Logprobs give a continuous uncertainty signal per token without requiring multiple samples. A model that assigns probability 0.3 to "True", 0.25 to "False", 0.2 to "None" has high token entropy even though greedy decoding deterministically outputs "True" — this completely eliminates the zero-entropy floor effect (71% of candidates scoring zero). Logprobs also capture uncertainty at a finer granularity: even within a single predicted output, some tokens may be high-confidence (boilerplate structure) while others are low-confidence (the actual computed value), and the signal comes from the latter.

**Requirement:** Logprob access via API. Fireworks API supports logprobs for most models. Need to verify which models in our set expose top-k logprobs and at what k.

#### Estimator C: Verbalized Confidence

```python
def information_gain_verbalized(model, context, candidate_test):
    """Ask the model to explicitly rate its uncertainty."""
    prompt = f"""Given this function and test history, answer two questions:
    1. Rate 0-100: How confident are you that you can predict the exact output?
    2. Rate 0-100: How likely is this test to hit branches not covered by previous tests?
    Respond with just the two numbers."""
    response = model.generate(context + prompt + candidate_test)
    output_confidence, branch_confidence = parse_scores(response)
    # Lower confidence = higher information gain
    return (100 - output_confidence) / 100
```

**Why this might work:** LLMs may have better-calibrated *verbalized* uncertainty than *sampling* uncertainty. A model can correctly state "I'm 30% confident about this output" even when its sampling process deterministically produces one answer.

#### Estimator D: Hybrid (Multi-Model + Logprobs + Verbalized)

Combine signals: use multi-model disagreement as the primary score, weighted by mean logprob entropy and/or verbalized confidence. Models that have high token-level entropy on their prediction AND disagree with other models contribute more to the information gain estimate.

#### Previous Estimator (deprecated for now): Single-Model Sampling Entropy

```python
def information_gain_sampling(model, context, candidate_test, S=10):
    predictions = [
        model.predict_output(context, candidate_test)
        for _ in range(S)
    ]
    clusters = cluster_predictions(predictions)
    probs = [len(c) / S for c in clusters]
    return -sum(p * log(p) for p in probs if p > 0)
```

This was the original estimator from Plan v3. Phase 1 calibration showed it fails on realistic code — see Section 6 for full diagnosis.

### 5.2 Core Loop

```
CURIOSITY_GUIDED_TESTING(program P, models M[], budget N, K=5):

  context = {source: P, tests: [], coverage: empty}

  for step in range(N):
    # Generate K candidate test inputs (use one model or rotate)
    candidates = M[0].generate_test_candidates(context, K)

    # Score each by information gain (using selected estimator)
    for c in candidates:
        c.score = information_gain(M, context, c)  # ensemble, verbalized, or hybrid

    # Select and execute highest-scoring candidate
    best = argmax(candidates, key=score)
    result = execute(P, best)

    # Update context (this IS the world model update: h → h')
    context.tests.append((best, result))
    context.coverage.update(result.branches_hit)

  return context.coverage
```

### 5.3 Planning-Aware Extension (Curiosity Q-Values)

From Sun et al. Proposition 1:

```
q(test | context) = ḡ(test | context) + γ · E_result[v(context')]
```

where v(context') = max over future tests of their information gain from the updated context. Approximate with 1-step lookahead:

```python
def curiosity_q_value(models, context, test, gamma=0.5):
    immediate = information_gain(models, context, test)

    # Simulate executing the test
    predicted_result = models[0].predict_result(context, test)
    context_prime = context.with_test(test, predicted_result)

    # Estimate best future info gain from updated context
    future_tests = models[0].generate_test_candidates(context_prime, K=3)
    future_value = max(information_gain(models, context_prime, ft) for ft in future_tests)

    return immediate + gamma * future_value
```

---

## 6. What We've Learned So Far (Phase 0–1 Results)

### Phase 0: Toy Programs (Completed ✓)

Output prediction entropy worked on toy programs: Spearman ρ = 0.3–0.7, curiosity matched or beat greedy coverage. This validated the basic framework.

### Phase 1: ULT Calibration (Completed ✓)

**Estimator 1 — Single-model sampling entropy (S=6, then S=20):**

| Model | Overall rho | 20+ Complexity rho (S=6) | 20+ Complexity rho (S=20) | Verdict |
|---|---|---|---|---|
| Gemini 3 Flash | 0.081 | 0.148 | 0.054 | PIVOT |
| Mistral Large | -0.089 | -0.053 | 0.095 | PIVOT |

Did not meet GO threshold (ρ > 0.15). Increasing S from 6 to 20 made Gemini *worse* (0.148 → 0.054), suggesting the S=6 result was partially noise. Mistral improved but remained below threshold.

**Estimator 2 — Coverage-prediction disagreement (S=20, complexity 20+):**

| Model | Spearman rho | Verdict |
|---|---|---|
| Gemini 3 Flash | -0.289 | STOP |
| Mistral Large | -0.106 | STOP |

Anti-correlated. Models unanimously predict "yes, new branches" for every candidate (Gemini mean disagreement = 0.000 at state 0). LLMs are overconfident about coverage predictions.

### Diagnosis: Why the Estimators Failed

**The world model (LLM + history) is sound.** The LLM does update its predictions as test history accumulates — in-context learning works as the Bayesian update mechanism.

**The uncertainty readout is broken.** Single-model temperature sampling does not express calibrated epistemic uncertainty about program behavior. The sampling distribution mixes linguistic variation, token-level randomness, and genuine epistemic uncertainty in ways that don't correlate with actual information gain. Key evidence:

1. **Zero-entropy floor:** Gemini produces identical predictions across 20 samples 71% of the time. This isn't certainty — it's that the decoding process is peaked regardless of the model's actual uncertainty.

2. **Model-specificity:** Gemini shows weak positive correlation; Mistral shows negative correlation. If sampling entropy were tracking a real quantity (information gain), it wouldn't flip sign across models.

3. **State interaction:** Both models show positive correlation at state 0 (fresh), which inverts at state 5+. The estimator works when everything is uncertain and fails when discrimination matters most.

### Diagnosis: The Oracle Ceiling

**At state 0 (fresh):** Best/avg ratio among K=10 candidates is ~1.1x. All candidates are roughly equivalent — the LLM generates similar-quality tests regardless of selection. A perfect oracle only adds ~10% coverage. **Candidate selection is the wrong intervention point at state 0.**

**At states 5+ (after exploration):** Best/avg ratio jumps to 2–4x — selection matters in relative terms, but absolute gains are tiny (best ≈ 2–3 branches, avg ≈ 1 branch).

**Overall:** A perfect oracle ranker adds ~10% total coverage. The main bottleneck at state 0 is candidate *diversity*, not candidate *selection*.

### Why Toy Programs Worked but ULT Didn't

Four conditions present in toy programs and absent in ULT:

1. **Distinct behavioral regions:** Toy programs produce qualitatively different outputs (None vs. dict vs. error string). ULT functions produce similar-looking outputs (numbers, lists). Sampling entropy correlates with behavioral uncertainty only when outputs are structurally diverse.

2. **Diverse candidate pools:** Simple input spaces → LLM generates varied candidates. Complex input spaces (class instances, nested structures) → LLM generates similar candidates.

3. **Moderate branch counts:** Toy programs have 8–40 arcs; plenty left to discover at state 5. ULT functions have 100+ arcs; coverage saturates fast, leaving sparse gains.

4. **Clear corridor structure:** Toy programs have explicit validation-then-logic structure. ULT's high-complexity functions are algorithmically complex but behaviorally homogeneous.

**The signal is real — it just doesn't transfer from toy programs to real-world functions via single-model sampling entropy.** The question is whether alternative uncertainty readouts can extract it.

---

## 7. What We're NOT Doing

**This is not about reward.** There's no reward signal being backpropagated. Information gain is a **test selection heuristic**, not a training objective. The metric (coverage) is measured externally by coverage tools, not optimized during search.

**This is not MCTS.** We don't do rollouts or backpropagation. It's sequential best-first selection guided by an information-theoretic acquisition function. Closer to Bayesian experimental design than tree search.

**This is not about getting the right answer.** Unlike math reasoning (where curiosity would need to improve accuracy), here exploration IS the objective. More coverage = more discovered behavior = success.

**We are not building a separate world model.** The LLM + history in its context window IS the world model. In-context learning IS the Bayesian update. The problem is reading out its uncertainty, not replacing it.

---

## 8. Experimental Design

### 8.1 Benchmarks

**Primary: TestGenEval (1,210 code-test file pairs, 11 Python repositories)**

From Meta Research (Jain, Synnaeve, Rozière), based on SWE-Bench. File-level test generation across real-world, well-maintained Python projects. GPT-4o achieves only 35.2% average coverage — massive headroom. Measures both coverage and mutation score (18.8% for GPT-4o). TestForge (2025) recently achieved 84.3% pass@1 using an agentic approach but coverage remains low for complex files. A lite version (160 pairs) exists for fast iteration.

**Why this is ideal for us:** 35% coverage with GPT-4o means the benchmark is genuinely hard and far from saturated. The file-level complexity (average 58 methods per file) creates natural corridor structure — you must understand module dependencies, class hierarchies, and initialization sequences before reaching deep test targets. Existing methods struggle specifically because they reason greedily about individual coverage gaps rather than planning exploration sequences.

**Secondary: ULT (3,909 Python functions, high cyclomatic complexity)**

From Huang et al. (2025). Specifically curated for high cyclomatic complexity with rigorous decontamination. On functions with cyclomatic complexity 10-20, models achieve <30% branch coverage, compared to ~70% on simpler benchmarks at matched complexity. This 40-point gap isolates the effect of structural complexity.

**Why this complements TestGenEval:** ULT is function-level (simpler setup, faster iteration) and explicitly selects for corridor-heavy functions. If curiosity-guided exploration helps most on ULT's complex functions, that directly validates the corridor hypothesis. Use ULT for Phase 0b calibration and fast prototyping, TestGenEval for the main paper results.

**Calibration: CRUXEval-O (800 functions)**

Use CRUXEval output prediction accuracy to validate our entropy estimator. Functions where the LLM is bad at predicting outputs should have high output entropy and should be exactly where curiosity-guided exploration should help most. Not a primary evaluation benchmark (likely saturated on frontier models), but useful for validating the information gain proxy.

### 8.2 Baselines

| Method | Description | Source |
|---|---|---|
| **Pynguin** | Search-based test generation (evolutionary) | Existing tool |
| **CodaMosa** | Pynguin + LLM when plateau detected | ICSE 2023 |
| **CoverUp** | Greedy coverage-gap-directed LLM generation | FSE 2025 |
| **TestForge** | Agentic LLM with file editing, test running, coverage reading | 2025 (same group as TestGenEval) |
| **LLM random** | LLM generates tests with no feedback | Our implementation |
| **LLM + coverage feedback** | LLM sees coverage gaps, generates targeted tests | Our implementation |
| **Curiosity greedy (ours)** | Select by information gain estimator (γ=0) | Our implementation |
| **Curiosity planning (ours)** | Select by curiosity Q-value (γ > 0) | Our implementation |
| **Random from candidates** | Same K candidates as curiosity, pick randomly | Ablation |
| **Coverage oracle** | Run ALL candidates, pick highest actual coverage gain | Upper bound |

**Note on compute matching:** TestGenEval measures coverage from a single generation pass (not iterative). Our method is inherently iterative (generate → score → select → execute → repeat). For fair comparison, we report both: (a) coverage at matched API cost, and (b) coverage at matched number of generated test functions.

### 8.3 Metrics

**Primary:** Branch coverage at fixed interaction budget (N = 10, 20, 50, 100 tests)

**Secondary:**
- Line coverage at fixed budget
- Mutation score (how many seeded bugs detected)
- Coverage AUC (area under coverage-vs-steps curve)
- Time to X% coverage
- Deep branch coverage (branches nested ≥3 conditions deep)

**Diagnostic:**
- Spearman correlation: information gain estimate vs. actual coverage gain
- Coverage improvement stratified by cyclomatic complexity
- Corridor traversal analysis on annotated functions
- Oracle ceiling (best/avg ratio) per function — characterizes where selection can help

### 8.4 Ablations

| Ablation | What it tests |
|---|---|
| Estimator: multi-model vs. logprobs vs. verbalized vs. hybrid vs. sampling entropy | Which uncertainty readout works? |
| Number of models in ensemble: 2, 3, 5 | Minimum ensemble size for useful disagreement |
| K (candidates per step): 3, 5, 10 | Breadth of candidate pool |
| γ (discount): 0, 0.3, 0.5, 0.7 | Value of planning ahead (γ=0 is greedy) |
| Code visible vs. hidden | White-box (practical) vs. black-box (scientific) |

### 8.5 Two Experimental Conditions

**White-box (code visible):** The LLM sees source code. Practical contribution — better than existing tools. Information gain captures "what will the runtime behavior be?" even when you can read the code.

**Black-box (code hidden):** The LLM sees only the function signature and accumulated test results. Scientific contribution — purest test of Bayesian exploration. The LLM must learn program behavior entirely from interaction. Closest to Sun et al.'s unknown-MDP setting.

---

## 9. Phases & Timeline

### Phase 0b: Uncertainty Readout Diagnostic (Day 1–3) ← YOU ARE HERE

**Goal:** Determine whether LLMs have useful uncertainty about program behavior that can be read out through alternative methods, before committing to a full estimator rebuild.

**Setup:** 30 functions from ULT 20+ complexity subset. Reuse existing candidates and ground-truth coverage gains from Phase 1. No new test execution — only new API calls for predictions.

**Test 1 — Verbalized Confidence:**
For each of the 30 functions × 10 candidates, prompt Gemini: "Given this function and test history, rate your confidence 0–100 that you can predict the exact output. Then rate 0–100 how likely this test is to hit branches not covered by previous tests." Measure Spearman ρ between these scores and actual coverage gain.

**Test 2 — Token-Level Logprob Entropy:**
For each of the 30 functions × 10 candidates, generate a single output prediction from Gemini (or whichever model supports logprobs via Fireworks) with logprobs enabled. Compute mean per-token entropy across the predicted output tokens. Measure Spearman ρ against actual coverage gain. This requires verifying logprob access — check which models in our Fireworks set expose top-k logprobs before running.

**Test 3 — Multi-Model Disagreement:**
For each of the 30 functions × 10 candidates, get a single output prediction from 3–5 different models (Gemini, Mistral, + 1–3 additional via Fireworks: Qwen, Llama, DeepSeek). Compute disagreement as entropy over distinct predictions across models. Measure Spearman ρ against actual coverage gain.

**Test 4 — Hybrid:**
Combine signals: each model provides a prediction (and logprobs if available) plus a verbalized confidence score. Compute weighted disagreement — models with high token entropy and low verbalized confidence contribute more. Measure Spearman ρ.

**Cost:** ~2,000 API calls (30 functions × 10 candidates × ~6 calls per candidate across tests). Estimated ~$5–8, <2 hours.

**Go/no-go (apply to each estimator independently):**
- **GO** if: Spearman ρ > 0.15 on 20+ complexity functions overall
- **STRONG GO** if: ρ > 0.15 overall AND ρ increases with complexity (replicating the trend from Phase 1 Gemini, but above threshold)
- **PIVOT** if: Weak correlation (0.05 < ρ < 0.15) → investigate contrastive prediction or embedding-based approaches
- **STOP** if: ρ < 0.05 or negative for ALL four estimators → LLMs lack useful uncertainty about program behavior; write up negative result

**What each outcome means:**

| Verbalized | Logprobs | Multi-Model | Interpretation | Next step |
|---|---|---|---|---|
| GO | GO | GO | LLMs have useful uncertainty; multiple readouts work | Use cheapest GO estimator, proceed to Phase 1b |
| FAIL | GO | * | Token-level uncertainty is calibrated; sampling readout was the bottleneck | Use logprob estimator (single model, cheap), proceed to Phase 1b |
| FAIL | FAIL | GO | Uncertainty exists across model population but not within any single model | Use multi-model ensemble (3–5x cost), proceed to Phase 1b |
| GO | FAIL | FAIL | Models can report uncertainty verbally but neither internal representations nor cross-model disagreement tracks it | Use verbalized confidence, proceed cautiously to Phase 1b |
| FAIL | FAIL | FAIL | LLMs lack calibrated uncertainty about program behavior at any readout level | Write up negative result (Path C) |

### Phase 1b: Full Calibration with Winning Estimator (Week 1–2)

**Only proceed here if Phase 0b returns at least one GO.**

Repeat the Phase 1 calibration protocol with the winning estimator from Phase 0b:

1. Run on all 97 ULT functions (not just the 30 from Phase 0b), stratified by complexity (10–15, 15–20, 20+)
2. Measure at states 0, 5, 10 — verify the new estimator doesn't degrade at later states like sampling entropy did
3. Compute oracle ceiling per function — characterize which functions have meaningful selection effects
4. Re-run on toy programs to verify the new estimator reproduces Phase 0 results (ρ = 0.3–0.7)

**Go/no-go:**
- **GO** if: Spearman ρ > 0.15 overall, AND does not invert at states 5/10, AND toy program results hold
- **PIVOT** if: Correlation holds at state 0 only → focus experimental claims on early exploration
- **STOP** if: Does not replicate

**Deliverable:** Calibration report v2. Validated estimator ready for Phase 2.

### Phase 2: Main Experiment on TestGenEval (Week 2–5)

Run curiosity-guided testing on TestGenEval:

1. Set up TestGenEval's Docker-based evaluation infrastructure (based on SWE-Bench)
2. Start with TestGenEval-Lite (160 pairs) for fast iteration, then scale to full benchmark
3. Run baselines: direct LLM generation (GPT-4o), CodaMosa (upgraded to GPT-4o), CoverUp, TestForge
4. Run curiosity greedy (γ=0) at matched compute budgets
5. Compare coverage and mutation score across methods

**Key comparisons:**
- Against TestForge (agentic approach, same benchmark group) — our closest competitor
- Against CoverUp (greedy coverage-guided) — tests whether curiosity beats greedy
- Stratified by file complexity — expect biggest gains on complex files with deep dependencies
- Stratified by oracle ceiling — show curiosity helps most where selection matters (best/avg > 1.5x)

**Expected results:** GPT-4o baseline achieves ~35% coverage. If curiosity-guided reaches ~45-50%, that's a substantial result. Even reaching 40% (a 14% relative improvement) is publishable given how hard this benchmark is.

**Deliverable:** Main comparison tables and scaling curves.

### Phase 3: Corridor Analysis on ULT + TestGenEval (Week 5–7)

1. Run curiosity planning (γ > 0) on both benchmarks
2. On ULT: stratify by cyclomatic complexity — planning-ahead should help most on highest-complexity functions (deepest corridors)
3. On TestGenEval: identify files where greedy curiosity plateaus but oracle shows higher coverage is reachable — these are the corridor files
4. Compare greedy (γ=0) vs. planning-aware (γ > 0) specifically on corridor-heavy subset
5. Case studies: walk through specific files showing corridor traversal (e.g., "curiosity invested 3 tests setting up fixtures before reaching the deep branch")

**Deliverable:** Corridor analysis showing planning-ahead value is concentrated on structurally complex code.

### Phase 4: Black-Box Condition & Paper (Week 7–10)

1. Re-run key experiments on ULT in code-hidden mode (black-box) — LLM sees only function signature + accumulated test results
2. Run ablations (estimator variants, ensemble size, K, γ)
3. Write paper

**Deliverable:** Complete paper draft.

---

## 10. Compute Budget

| Phase | API calls | Est. cost |
|---|---|---|
| ~~Phase 0: PoC~~ | ~~2K~~ | ~~$2~~ (completed) |
| ~~Phase 0.5: Infrastructure~~ | ~~5K~~ | ~~$3~~ (completed) |
| ~~Phase 1: ULT calibration~~ | ~~50K~~ | ~~$21~~ (completed, $12.69 + $8.72) |
| **Phase 0b: Uncertainty readout diagnostic** | **~2K** | **~$5–8** |
| Phase 1b: Full calibration with winning estimator | ~30K | ~$15–25 |
| Phase 2: TestGenEval-Lite (160 pairs, all methods) | ~200K | ~$60 |
| Phase 2: TestGenEval full (1,210 pairs, key methods) | ~400K | ~$120 |
| Phase 3: Planning-aware + corridor analysis | ~200K | ~$60 |
| Phase 4: Black-box + ablations | ~150K | ~$50 |
| **Total remaining** | **~980K** | **~$310–320** |
| **Already spent** | **~57K** | **~$27** |

Multi-model ensemble estimator costs 3–5x per candidate scoring compared to single-model. Budget above assumes 5-model ensemble for Phases 2–4. If verbalized confidence wins in Phase 0b (single model), costs drop significantly.

All API-based. Does not compete with A100s used for GRPO training in Plan v4.

---

## 11. Risk Register

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| ~~Entropy doesn't correlate with coverage gain~~ | ~~25%~~ | ~~Fatal~~ | **Realized.** Single-model sampling entropy failed calibration. Now testing alternative readouts in Phase 0b. |
| All uncertainty readouts fail (Phase 0b) | 35% | Fatal | If all three estimators fail, LLMs lack calibrated uncertainty about program behavior. Write up as negative result (still publishable as a workshop paper or negative result track). |
| Multi-model ensemble too expensive | 20% | Medium | If ensemble wins, optimize: test 2–3 models instead of 5, or use cheaper models. Budget allows 5-model ensemble but 3 may suffice. |
| Estimator works on ULT but not TestGenEval | 25% | High | TestGenEval's file-level complexity creates natural corridor structure → should help. Phase 1b validates before committing $60+ to Phase 2. |
| Curiosity ≈ baselines on TestGenEval | 25% | High | TestGenEval's difficulty (35% baseline) gives us room. Show advantage stratified by complexity and oracle ceiling. |
| Oracle ceiling too low (selection doesn't matter) | 30% | High | Already know ceiling is ~10% at state 0 for ULT. TestGenEval file-level tasks may have higher ceiling due to more diverse candidate space. Measure oracle ceiling on TestGenEval before full runs. |
| Planning-ahead adds nothing over greedy | 35% | Medium | Still publishable with greedy curiosity. Corridor analysis becomes "where planning would help." |
| LLM generates invalid test inputs | 30% | Medium | Retry with feedback; filter by syntax check; count valid tests only. Existing tools face the same problem. |
| TestGenEval Docker setup too complex | 25% | Medium | Start with TestGenEval-Lite (160 pairs). Fall back to ULT (function-level, simpler setup) as primary if Docker issues persist. |

---

## 12. Three Nested Claims (Updated)

**Claim 1 (revised, confidence depends on Phase 0b):** LLM uncertainty — measured through multi-model disagreement, verbalized confidence, or both — is a useful signal for test selection. Tests where the LLM ensemble disagrees about the output tend to discover more new branches.

- If Phase 0b succeeds: confidence ~60% (signal exists, estimator captures it)
- If Phase 0b fails: Claim 1 becomes a negative result (LLMs lack calibrated uncertainty for this task)

**Claim 2 (moderate confidence, ~40%):** Curiosity-guided selection outperforms greedy coverage feedback on complex functions under fixed budget. Strongest on functions with high oracle ceiling (best/avg > 1.5x) and in the black-box condition.

**Claim 3 (lower confidence, ~25%):** Planning-aware curiosity Q-values outperform greedy curiosity on corridor-heavy functions, by valuing validation-passing inputs for their future information content.

**The paper is publishable with Claim 1 alone** — either as a positive result (new signal for test selection) or as a characterized negative result (LLM uncertainty doesn't transfer to Bayesian information gain, here's why, here's the boundary).

---

## 13. Paper Outline

**Title:** "Planning to Explore: Curiosity-Guided Test Generation via Bayesian Exploration"

1. **Introduction:** LLM test generation uses greedy coverage feedback; we ground it in optimal Bayesian exploration theory. This enables planning through coverage corridors.

2. **Background:** Sun et al. 2011 (curiosity Q-values, non-additivity, corridor problem). LLM test generation landscape (CodaMosa, CoverUp, TestPilot). Schmidhuber 1990–2011 artificial curiosity lineage.

3. **Method:** Program exploration as Bayesian exploration. LLM + history as world model. Uncertainty readout methods (multi-model disagreement, token-level logprob entropy, verbalized confidence). Greedy and planning-aware algorithms.

4. **Experiments:** TestGenEval (primary), ULT (corridor analysis), CRUXEval (calibration). Baselines: CodaMosa, CoverUp, TestForge. White-box and black-box conditions.

5. **Results:** Coverage and mutation score comparisons, scaling curves, complexity-stratified analysis, corridor case studies. Oracle ceiling analysis showing where selection matters.

6. **Analysis:** When curiosity helps (complex functions, high oracle ceiling, small budgets, black-box). Calibration validation. Why single-model sampling entropy fails and multi-model disagreement succeeds (or: why all readouts fail, if negative result). Characterization of program properties that predict framework applicability.

7. **Related Work:** LLM test generation (CodaMosa, CoverUp, ChatUniTest, TestPilot, EvoGPT), Bayesian exploration (Sun et al., Schmidhuber), information-gain-guided LLMs (UoT, BED-LLM, CuriosiTree, AutoDiscovery), intrinsic motivation for LLM training (i-MENTOR, MERCI), LLM uncertainty estimation and calibration.

8. **Conclusion.**

**Target venues:** ICSE 2027, FSE 2027 (SE framing) or NeurIPS 2026 (ML framing).

---

## 14. Relationship to Existing Work

### What exists
- **CoverUp:** Greedy — tells the LLM which lines lack coverage, asks for tests targeting those lines. Reactive.
- **CodaMosa:** Invokes the LLM when evolutionary search stalls. Reactive.
- **TestForge:** Agentic — gives the LLM tools to edit files, run tests, read coverage reports. Most sophisticated existing approach but still greedy (targets whatever coverage gap it sees next).
- **UoT / BED-LLM / CuriosiTree:** Use information gain for LLM question-asking, but for *external* information acquisition (asking oracles), not internal program exploration.
- **i-MENTOR / MERCI:** Intrinsic motivation during RL *training*, not during *inference/exploration*.

### What's new
1. First application of optimal Bayesian exploration to LLM test generation
2. Multi-model disagreement as information gain proxy (or: analysis of why LLM uncertainty fails as Bayesian proxy)
3. Planning-aware test selection through coverage corridors (no existing tool does this)
4. Black-box program exploration via information gain (no existing tool operates without coverage feedback)
5. Oracle ceiling analysis characterizing when candidate selection matters for LLM test generation
6. Direct instantiation of Sun et al.'s / Schmidhuber's framework in a modern LLM setting with automatic evaluation

---

## 15. Relationship to Plan v4 (Agent ETE)

This is a **parallel track**, not a replacement.

| | Plan v4 (Agent ETE) | This plan (Code Exploration) |
|---|---|---|
| Setting | Agent in text games | LLM generating tests |
| Metric | Task completion accuracy | Branch coverage (exploration) |
| Schmidhuber era | 1990/1991 curiosity signals | 2011 optimal Bayesian exploration |
| Environment | ScienceWorld / Jericho | Python programs |
| Risk of "no improvement" | High (accuracy metric) | Lower (exploration metric) |
| Compute | A100 for GRPO | API calls only |

If both work, they form a two-paper thesis story: "Schmidhuber's curiosity framework for LLMs — from training (Plan v4) to inference (this plan)."

---

## 16. Decision Points

**After Phase 0b (Day 3):** Does any uncertainty readout (verbalized, logprobs, multi-model, hybrid) produce ρ > 0.15 on ULT 20+ complexity?
- YES → proceed to Phase 1b with winning estimator
- WEAK (0.05–0.15) → investigate contrastive prediction or embedding-based readouts before proceeding
- NO (all four fail) → write up negative result, refocus effort on Plan v4 or generation diversity research

**After Phase 1b (Week 2):** Does the winning estimator hold across all complexity tiers and exploration states? Does it reproduce toy program results?
- YES → proceed to Phase 2
- PARTIAL (works at state 0 only) → scope claims to early exploration, proceed cautiously
- NO → escalate to STOP decision

**After Phase 2 (Week 5):** Does curiosity beat baselines on TestGenEval? Even a relative improvement from 35% → 40% coverage is significant on this benchmark. If no improvement, check ULT results (function-level may be easier) and black-box condition.

**After Phase 3 (Week 7):** Does planning-ahead help on ULT's highest-complexity functions and TestGenEval's hardest files? If yes, full theoretical story. If no, publish with greedy curiosity + theoretical framework.