# Curiosity-Guided Code Exploration
## Optimal Bayesian Exploration for LLM-Based Test Generation

### Final Research Plan — March 2026

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

The core signal: **output prediction entropy**. Before running a test, ask the LLM "what will this program return for this input?" multiple times. High disagreement across predictions = high uncertainty = informative test.

```python
def information_gain(model, context, candidate_test, S=10):
    # Sample S predictions of what P(x) will produce
    predictions = [
        model.predict_output(context, candidate_test)
        for _ in range(S)
    ]
    # Cluster semantically equivalent predictions
    clusters = cluster_predictions(predictions)
    # Compute entropy
    probs = [len(c) / S for c in clusters]
    return -sum(p * log(p) for p in probs if p > 0)
```

**Why this works:** If the LLM already knows what P(x) will do, entropy is low — uninformative test. If the LLM is genuinely uncertain, entropy is high — informative test. This maps directly to ḡ(a|h) = I(O; Θ|h,a) from Sun et al.

**CRUXEval connection:** CRUXEval-O measures exactly this capability — output prediction. Functions where the LLM scores poorly on CRUXEval-O are functions where output entropy will be high, which is where curiosity-guided exploration should help most. We can use this to validate the entropy estimator.

### 5.2 Core Loop

```
CURIOSITY_GUIDED_TESTING(program P, model M, budget N, K=5, S=10):

  context = {source: P, tests: [], coverage: empty}

  for step in range(N):
    # Generate K candidate test inputs
    candidates = M.generate_test_candidates(context, K)

    # Score each by output prediction entropy
    for c in candidates:
        c.score = information_gain(M, context, c, S)

    # Select and execute highest-entropy candidate
    best = argmax(candidates, key=score)
    result = execute(P, best)

    # Update context
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
def curiosity_q_value(model, context, test, gamma=0.5):
    immediate = information_gain(model, context, test)

    # Simulate executing the test
    predicted_result = model.predict_result(context, test)
    context_prime = context.with_test(test, predicted_result)

    # Estimate best future info gain from updated context
    future_tests = model.generate_test_candidates(context_prime, K=3)
    future_value = max(information_gain(model, context_prime, ft) for ft in future_tests)

    return immediate + gamma * future_value
```

---

## 6. What We're NOT Doing

**This is not about reward.** There's no reward signal being backpropagated. Information gain is a **test selection heuristic**, not a training objective. The metric (coverage) is measured externally by coverage tools, not optimized during search.

**This is not MCTS.** We don't do rollouts or backpropagation. It's sequential best-first selection guided by an information-theoretic acquisition function. Closer to Bayesian experimental design than tree search.

**This is not about getting the right answer.** Unlike math reasoning (where curiosity would need to improve accuracy), here exploration IS the objective. More coverage = more discovered behavior = success.

---

## 7. Experimental Design

### 7.1 Benchmarks

**Primary: TestGenEval (1,210 code-test file pairs, 11 Python repositories)**

From Meta Research (Jain, Synnaeve, Rozière), based on SWE-Bench. File-level test generation across real-world, well-maintained Python projects. GPT-4o achieves only 35.2% average coverage — massive headroom. Measures both coverage and mutation score (18.8% for GPT-4o). TestForge (2025) recently achieved 84.3% pass@1 using an agentic approach but coverage remains low for complex files. A lite version (160 pairs) exists for fast iteration.

**Why this is ideal for us:** 35% coverage with GPT-4o means the benchmark is genuinely hard and far from saturated. The file-level complexity (average 58 methods per file) creates natural corridor structure — you must understand module dependencies, class hierarchies, and initialization sequences before reaching deep test targets. Existing methods struggle specifically because they reason greedily about individual coverage gaps rather than planning exploration sequences.

**Secondary: ULT (3,909 Python functions, high cyclomatic complexity)**

From Huang et al. (2025). Specifically curated for high cyclomatic complexity with rigorous decontamination. On functions with cyclomatic complexity 10-20, models achieve <30% branch coverage, compared to ~70% on simpler benchmarks at matched complexity. This 40-point gap isolates the effect of structural complexity.

**Why this complements TestGenEval:** ULT is function-level (simpler setup, faster iteration) and explicitly selects for corridor-heavy functions. If curiosity-guided exploration helps most on ULT's complex functions, that directly validates the corridor hypothesis. Use ULT for Phase 1 calibration and fast prototyping, TestGenEval for the main paper results.

**Calibration: CRUXEval-O (800 functions)**

Use CRUXEval output prediction accuracy to validate our entropy estimator. Functions where the LLM is bad at predicting outputs should have high output entropy and should be exactly where curiosity-guided exploration helps most. Not a primary evaluation benchmark (likely saturated on frontier models), but useful for validating the information gain proxy.

### 7.2 Baselines

| Method | Description | Source |
|---|---|---|
| **Pynguin** | Search-based test generation (evolutionary) | Existing tool |
| **CodaMosa** | Pynguin + LLM when plateau detected | ICSE 2023 |
| **CoverUp** | Greedy coverage-gap-directed LLM generation | FSE 2025 |
| **TestForge** | Agentic LLM with file editing, test running, coverage reading | 2025 (same group as TestGenEval) |
| **LLM random** | LLM generates tests with no feedback | Our implementation |
| **LLM + coverage feedback** | LLM sees coverage gaps, generates targeted tests | Our implementation |
| **Curiosity greedy (ours)** | Select by output prediction entropy | Our implementation |
| **Curiosity planning (ours)** | Select by curiosity Q-value (γ > 0) | Our implementation |
| **Random from candidates** | Same K candidates as curiosity, pick randomly | Ablation |
| **Coverage oracle** | Run ALL candidates, pick highest actual coverage gain | Upper bound |

**Note on compute matching:** TestGenEval measures coverage from a single generation pass (not iterative). Our method is inherently iterative (generate → score → select → execute → repeat). For fair comparison, we report both: (a) coverage at matched API cost, and (b) coverage at matched number of generated test functions.

### 7.3 Metrics

**Primary:** Branch coverage at fixed interaction budget (N = 10, 20, 50, 100 tests)

**Secondary:**
- Line coverage at fixed budget
- Mutation score (how many seeded bugs detected)
- Coverage AUC (area under coverage-vs-steps curve)
- Time to X% coverage
- Deep branch coverage (branches nested ≥3 conditions deep)

**Diagnostic:**
- Spearman correlation: output prediction entropy vs. actual coverage gain
- Coverage improvement stratified by cyclomatic complexity
- Corridor traversal analysis on annotated functions

### 7.4 Ablations

| Ablation | What it tests |
|---|---|
| S (prediction samples): 3, 5, 10, 20 | How many samples for reliable entropy? |
| K (candidates per step): 3, 5, 10 | Breadth of candidate pool |
| γ (discount): 0, 0.3, 0.5, 0.7 | Value of planning ahead (γ=0 is greedy) |
| Code visible vs. hidden | White-box (practical) vs. black-box (scientific) |
| Output entropy vs. coverage-prediction disagreement | Which info gain estimator works best? |

### 7.5 Two Experimental Conditions

**White-box (code visible):** The LLM sees source code. Practical contribution — better than existing tools. Information gain captures "what will the runtime behavior be?" even when you can read the code.

**Black-box (code hidden):** The LLM sees only the function signature and accumulated test results. Scientific contribution — purest test of Bayesian exploration. The LLM must learn program behavior entirely from interaction. Closest to Sun et al.'s unknown-MDP setting.

---

## 8. Phases & Timeline

### Phase 0: Proof-of-Concept (Day 1–3)

**Script:** `phase0_poc.py` (already written)

Run the complete pipeline on 3 toy programs:
- `corridor_basic` — order processor with 4 validation gates (corridor)
- `nested_conditions` — triangle classifier with 3 gates (corridor)
- `flat_no_corridor` — extended fizzbuzz (control)

Three strategies (random, greedy coverage, curiosity-guided), 15-step budget each.

**GO/NO-GO criteria:**
- GO: Curiosity > Random on corridor programs AND Spearman ρ(entropy, coverage gain) > 0.1
- INVESTIGATE: Mixed signals → try alternative entropy estimators
- STOP: Entropy negatively correlates with coverage gain

**Cost:** ~$2, ~15 minutes.

### Phase 0.5: Infrastructure (Week 1)

**Only proceed here after Phase 0 returns GO.**

Build the production harness from the PoC scaffold:

```
curiosity_explorer/
  ├── runner/
  │   ├── sandbox.py           # Safe execution (subprocess, timeout, memory limit)
  │   ├── coverage.py          # Branch/line/arc coverage via coverage.py
  │   └── trace_parser.py      # Parse execution traces for LLM context
  ├── explorer/
  │   ├── candidate_gen.py     # LLM test input generation with retries
  │   ├── info_gain.py         # Output entropy + coverage-prediction + embedding proxy
  │   ├── curiosity_search.py  # Greedy and planning-aware strategies
  │   └── baselines.py         # Random, greedy coverage, oracle
  ├── benchmarks/
  │   ├── toy_programs.py      # Phase 0 programs (regression tests)
  │   ├── cruxeval/             # CRUXEval function loader
  │   └── testgeneval/          # TestGenEval Docker-based benchmark loader
  ├── analysis/
  │   ├── calibration.py       # Entropy-coverage correlation
  │   ├── corridor_analysis.py # Corridor detection and stratified results
  │   └── plotting.py          # Coverage curves, comparison tables
  └── run_experiment.py        # Main experiment driver
```

**Key improvements over PoC:**
- Proper sandboxing (subprocess with resource limits, not bare exec)
- Batch API calls for efficiency
- Response caching (same prompt → cached response, avoid redundant calls)
- Per-branch tracking (which specific branches are hit, not just arc count)
- Structured logging and result serialization for reproducibility
- TestGenEval and ULT benchmark loaders (Docker setup for TestGenEval's SWE-Bench environments)
- CRUXEval loader for calibration validation

**Validation:** Re-run Phase 0's toy programs through the production harness. Results should match PoC within noise.

**Deliverable:** Pipeline that can run any Python function through all strategies with full metrics.

### Phase 1: Calibration on ULT (Week 1–2)

Use ULT's high-complexity functions for fast calibration of the information gain estimator:

1. Select 100 functions from ULT, stratified by cyclomatic complexity (10-15, 15-20, 20+)
2. For each function, generate K=10 candidate tests at various exploration states (after 0, 5, 10 tests)
3. For each candidate: compute output prediction entropy AND actually run it to measure real coverage gain
4. Measure Spearman correlation between predicted entropy and actual coverage gain
5. Stratify by complexity: does the correlation hold more strongly on higher-complexity functions?

**Secondary calibration on CRUXEval-O:** Verify that functions where the LLM is bad at output prediction (low CRUXEval-O accuracy) have high output entropy in our estimator. This validates the proxy at scale without needing to run exploration.

**Go/no-go:**
- **GO** if: Spearman ρ > 0.15 on ULT functions overall, stronger on high-complexity subset
- **PIVOT** if: Weak correlation → try coverage-prediction disagreement as alternative estimator
- **STOP** if: Neither proxy correlates → fundamental issue

**Deliverable:** Calibration report. Entropy estimator validated or replaced.

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
2. Run ablations (S, K, γ, estimator variants)
3. Write paper

**Deliverable:** Complete paper draft.

---

## 9. Compute Budget

| Phase | API calls | Est. cost |
|---|---|---|
| Phase 0: PoC | ~2K | ~$2 |
| Phase 0.5: Infrastructure + validation | ~5K | ~$3 |
| Phase 1: ULT calibration (100 functions) | ~50K | ~$20 |
| Phase 2: TestGenEval-Lite (160 pairs, all methods) | ~200K | ~$60 |
| Phase 2: TestGenEval full (1,210 pairs, key methods) | ~400K | ~$120 |
| Phase 3: Planning-aware + corridor analysis | ~200K | ~$60 |
| Phase 4: Black-box + ablations | ~150K | ~$50 |
| **Total** | **~1M** | **~$315** |

TestGenEval's Docker-based execution is heavier than simple function-level testing — budget extra time for environment setup. TestGenEval-Lite exists specifically for fast iteration ($60 for full evaluation), then scale to the full benchmark for the final paper numbers.

All API-based. Does not compete with A100s used for GRPO training in Plan v4.

---

## 10. Risk Register

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| Entropy doesn't correlate with coverage gain | 25% | Fatal | Phase 0 tests this for $2. Phase 1 validates on ULT at scale. Try coverage-prediction disagreement as alternative. |
| Curiosity ≈ baselines on TestGenEval | 25% | High | TestGenEval's difficulty (35% baseline) gives us room. Show advantage stratified by complexity and on ULT's high-complexity subset. |
| Planning-ahead adds nothing over greedy | 35% | Medium | Still publishable with greedy curiosity. Corridor analysis becomes "where planning would help." |
| LLM generates invalid test inputs | 30% | Medium | Retry with feedback; filter by syntax check; count valid tests only. Existing tools face the same problem. |
| TestGenEval Docker setup too complex | 25% | Medium | Start with TestGenEval-Lite (160 pairs). Fall back to ULT (function-level, simpler setup) as primary if Docker issues persist. |
| TestForge is too strong a baseline | 20% | Medium | TestForge is agentic (edits files, runs tests iteratively). Our method is lighter-weight. Even matching TestForge with a simpler approach is a result. |

---

## 11. Three Nested Claims

**Claim 1 (high confidence, ~80%):** Output prediction entropy is a useful signal for test selection. Tests where the LLM is uncertain about the output tend to discover more new branches.

**Claim 2 (moderate confidence, ~50%):** Curiosity-guided selection outperforms greedy coverage feedback on complex functions under fixed budget. Strongest in the black-box condition where coverage feedback isn't available.

**Claim 3 (lower confidence, ~30%):** Planning-aware curiosity Q-values outperform greedy curiosity on corridor-heavy functions, by valuing validation-passing inputs for their future information content.

**The paper is publishable with Claim 1 alone** (new signal for test selection, validated on existing benchmarks). Claims 2 and 3 make it progressively stronger.

---

## 12. Paper Outline

**Title:** "Planning to Explore: Curiosity-Guided Test Generation via Bayesian Exploration"

1. **Introduction:** LLM test generation uses greedy coverage feedback; we ground it in optimal Bayesian exploration theory. This enables planning through coverage corridors.

2. **Background:** Sun et al. 2011 (curiosity Q-values, non-additivity, corridor problem). LLM test generation landscape (CodaMosa, CoverUp, TestPilot).

3. **Method:** Program exploration as Bayesian exploration. Output prediction entropy as information gain. Greedy and planning-aware algorithms.

4. **Experiments:** TestGenEval (primary), ULT (corridor analysis), CRUXEval (calibration). Baselines: CodaMosa, CoverUp, TestForge. White-box and black-box conditions.

5. **Results:** Coverage and mutation score comparisons, scaling curves, complexity-stratified analysis, corridor case studies.

6. **Analysis:** When curiosity helps (complex functions, small budgets, black-box). Calibration validation. Failure modes.

7. **Related Work:** LLM test generation (CodaMosa, CoverUp, ChatUniTest, TestPilot, EvoGPT), Bayesian exploration (Sun et al., Schmidhuber), information-gain-guided LLMs (UoT, BED-LLM, CuriosiTree, AutoDiscovery), intrinsic motivation for LLM training (i-MENTOR, MERCI).

8. **Conclusion.**

**Target venues:** ICSE 2027, FSE 2027 (SE framing) or NeurIPS 2026 (ML framing).

---

## 13. Relationship to Existing Work

### What exists
- **CoverUp:** Greedy — tells the LLM which lines lack coverage, asks for tests targeting those lines. Reactive.
- **CodaMosa:** Invokes the LLM when evolutionary search stalls. Reactive.
- **TestForge:** Agentic — gives the LLM tools to edit files, run tests, read coverage reports. Most sophisticated existing approach but still greedy (targets whatever coverage gap it sees next).
- **UoT / BED-LLM / CuriosiTree:** Use information gain for LLM question-asking, but for *external* information acquisition (asking oracles), not internal program exploration.
- **i-MENTOR / MERCI:** Intrinsic motivation during RL *training*, not during *inference/exploration*.

### What's new
1. First application of optimal Bayesian exploration to LLM test generation
2. Planning-aware test selection through coverage corridors (no existing tool does this)
3. Black-box program exploration via information gain (no existing tool operates without coverage feedback)
4. Direct instantiation of Sun et al.'s framework in a modern LLM setting with automatic evaluation

---

## 14. Relationship to Plan v4 (Agent ETE)

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

## 15. Decision Points

**After Phase 0 (Day 3):** Go / Investigate / Stop. Cost to reach this point: $2.

**After Phase 0.5 (Week 1):** Does the production harness reproduce Phase 0 results? If toy program results don't match, debug before scaling up.

**After Phase 1 (Week 2):** Is the entropy estimator validated on ULT's high-complexity functions? If Spearman ρ > 0.15, proceed with confidence. If marginal, switch to coverage-prediction disagreement.

**After Phase 2 (Week 5):** Does curiosity beat baselines on TestGenEval? Even a relative improvement from 35% → 40% coverage is significant on this benchmark. If no improvement, check ULT results (function-level may be easier) and black-box condition.

**After Phase 3 (Week 7):** Does planning-ahead help on ULT's highest-complexity functions and TestGenEval's hardest files? If yes, full theoretical story. If no, publish with greedy curiosity + theoretical framework.