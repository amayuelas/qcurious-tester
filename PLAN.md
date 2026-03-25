# Research Plan: Curiosity-Guided Inference-Time Search

## Experiment Plan v1 — March 2026

---

## 0. Executive Summary

**Claim:** Planning-aware information gain (curiosity Q-values from Sun et al. 2011) is a better objective than reward-based value functions for guiding inference-time tree search in LLM reasoning — especially on problems with "corridor" structure where intermediate steps are locally uninformative but globally necessary.

**Minimum publishable result:** On MATH-500 hard problems, curiosity-guided search matches or exceeds Tree-of-Thought and best-of-N baselines at equal compute budget, *without* requiring a trained verifier or access to ground truth during search.

**Timeline:** ~8 weeks to Phase 3 completion (minimum paper). ~12 weeks including Phase 4 (full paper).

**Key risk:** LLM answer-distribution entropy estimates are too noisy to produce meaningful information gain signals. Phase 1 exists entirely to test this.

---

## 1. Phase 0: Infrastructure & Calibration Validation (Week 1–2)

### Goal
Determine whether LLM sampling produces entropy estimates reliable enough to make information gain meaningful. This is the single make-or-break question. If it fails, the project pivots or stops.

### 1.1 Calibration Measurement

**Setup:**
- Select 200 problems from MATH-500 (stratified by difficulty level 1–5)
- For each problem, generate partial reasoning chains at depths d = {0, 1, 2, 3, 4} steps using a base model (Qwen2.5-Math-7B or similar via Fireworks)
- At each depth, sample S = {10, 20, 50} answer completions: "Given this reasoning so far, what is the final answer?"
- Record the empirical answer distribution and compute entropy

**Measurements:**
1. **Calibration plot**: When the model assigns empirical probability p̂ to the correct answer, is it actually correct p̂ of the time? Plot reliability diagram.
2. **Entropy-depth curve**: Does H(answer|chain) decrease monotonically with chain depth on average? (It should — deeper reasoning should narrow the answer space.)
3. **Entropy-correctness correlation**: Among completed chains, do chains with lower final entropy have higher accuracy? (Must be true for the approach to work.)
4. **Sample efficiency**: How does entropy estimate variance change with S? At what S does the estimate stabilize? Compute coefficient of variation across 5 independent runs.

**Go/no-go criterion:**
- **GO** if: Entropy decreases with depth on average AND entropy-correctness correlation ρ > 0.3 AND estimates stabilize at S ≤ 30
- **PIVOT** if: Entropy is flat or noisy across depth → try semantic entropy clustering (group equivalent answers) before giving up
- **STOP** if: Even with semantic entropy, no reliable signal

### 1.2 Semantic Entropy Implementation

Following Kuhn et al. (2023), cluster sampled answers by semantic equivalence before computing entropy. For math, this is straightforward: parse numerical answers and cluster by value (handling formatting variants like "42", "forty-two", "x = 42", "the answer is 42").

```python
# Pseudocode
def semantic_entropy(answer_samples):
    # Parse to canonical form
    canonical = [parse_math_answer(a) for a in answer_samples]
    # Cluster by value
    counts = Counter(canonical)
    probs = [c / len(canonical) for c in counts.values()]
    return -sum(p * log(p) for p in probs)
```

### 1.3 Infrastructure

- **answer_sampler.py**: Given (problem, partial_chain, depth), sample S completions and compute raw + semantic entropy
- **calibration_eval.py**: Run over MATH-500 subset, produce calibration plots
- **api_wrapper.py**: Fireworks API with batching, rate limiting, caching

**Compute estimate:** 200 problems × 5 depths × 50 samples = 50,000 API calls. At ~500 tokens/call, ~25M tokens. Cheap via Fireworks on a 7B model.

**Deliverable:** Calibration report with go/no-go decision. ~3 days of compute, 2 days analysis.

---

## 2. Phase 1: Greedy Information Gain Search (Week 2–4)

### Goal
Implement the simplest version of curiosity-guided search (greedy, τ=1) and compare against baselines on MATH-500. Establish whether information gain alone, without planning ahead, provides competitive search guidance.

### 2.1 Algorithm Implementation

```
GREEDY_CURIOSITY_SEARCH(problem, model, budget B, samples S, candidates K):

  tree = Tree(root=problem_statement)

  for step in range(B):
    for each leaf h in tree:
      # Current answer entropy
      H_current = semantic_entropy(sample_answers(model, h, S))

      # Score candidate next steps
      candidates = sample_reasoning_steps(model, h, K)
      for a in candidates:
        # Simulate executing step (in math, mostly deterministic: N=1)
        outcome = generate_step_outcome(model, h, a)
        H_after = semantic_entropy(sample_answers(model, h + a + outcome, S))
        info_gain[a] = H_current - H_after

    # Expand highest-gain (leaf, step) pair
    best_leaf, best_step = argmax(info_gain)
    tree.expand(best_leaf, best_step)

    # Early termination
    if min_leaf_entropy(tree) < epsilon:
      break

  return answer_from_lowest_entropy_leaf(tree)
```

**Key implementation decisions:**
- **N=1 for outcomes**: Math reasoning steps are nearly deterministic. One forward pass suffices to "execute" a step. This dramatically reduces cost.
- **K=5 candidates**: Sample 5 candidate next reasoning steps per leaf.
- **S=20 answer samples**: Based on Phase 0 calibration results (may adjust).
- **B=10 expansions**: Total compute budget per problem.
- **Pruning**: Only expand leaves with H > ε (skip already-confident leaves).

### 2.2 Baselines

All baselines get the same total compute budget (measured in forward passes):

| Method | Description | Forward passes per problem |
|---|---|---|
| **Best-of-N** | Generate N complete solutions, pick most common answer (self-consistency) | N full chains |
| **Best-of-N + self-eval** | Generate N solutions, LLM picks best | N + N eval calls |
| **Tree-of-Thought (ToT)** | Branch at each step, LLM self-evaluates to select | Matched to curiosity budget |
| **Greedy curiosity (ours)** | Branch at each step, select by information gain | K×B expansion + S×B entropy calls |
| **Random expansion** | Same tree structure, expand randomly | Same structure, no info gain |
| **Oracle (upper bound)** | Expand step that leads to correct answer | Requires ground truth |

**Compute matching**: The critical comparison is at *equal forward passes*. With K=5, S=20, B=10: ~(5 + 20) × 10 = 250 forward passes per problem. Best-of-N gets 250/avg_chain_length ≈ 50 complete solutions. ToT gets the same budget allocated to branching + evaluation.

### 2.3 Evaluation

**Dataset:** MATH-500 (full, all difficulty levels)

**Metrics:**
- **Accuracy** at fixed compute budget (primary)
- **Accuracy vs. compute** scaling curve (vary B from 1 to 50)
- **Accuracy stratified by difficulty** (expect biggest gains on Level 4–5)
- **Entropy reduction rate** per expansion step (diagnostic)
- **Corridor detection**: Qualitative analysis of problems where curiosity search takes "uninformative" intermediate steps that greedy baselines skip

**Deliverable:** Comparison table + scaling curves. If greedy curiosity ≥ ToT at equal compute, proceed to Phase 2. If greedy curiosity < ToT, analyze failure modes before proceeding.

### 2.4 Embedding Proxy Ablation

Simultaneously implement the cheap proxy:

```python
def info_gain_embed(model, encoder, h, a):
    outcome = generate_step_outcome(model, h, a)
    emb_before = encoder.encode(h)
    emb_after = encoder.encode(h + a + outcome)
    return 1 - cosine_similarity(emb_before, emb_after)
```

Using your existing sentence-transformer infrastructure. This reduces cost from ~25 forward passes per candidate to 1 LLM forward + 2 encoder forwards. Run the same evaluation and compare accuracy.

**Deliverable:** Ablation table showing exact entropy vs. embedding proxy vs. random. If embedding proxy retains >80% of exact entropy's accuracy gain, it becomes the default for subsequent phases.

---

## 3. Phase 2: Planning-Aware Search (Week 4–6)

### Goal
Implement the 1-step lookahead version (curiosity Q-values) and test whether it outperforms greedy, specifically on problems with corridor structure.

### 3.1 Curiosity Q-Value Approximation

The full recursive computation is intractable. Approximate with 1-step lookahead:

```
q(h, a) = ḡ(a|h) + γ · E_o[v(h·a·o)]
```

where v(h') estimates the future curiosity value of state h'.

**Three approximations for v(h'), in order of increasing cost:**

**(A) Diversity heuristic** (cheapest):
```python
def future_value_diversity(model, h_prime, K_future=3):
    # How many distinct productive continuations exist?
    future_steps = sample_reasoning_steps(model, h_prime, K_future)
    # If many diverse continuations → high future value
    # If stuck / repeating → low future value
    embeddings = [encoder.encode(s) for s in future_steps]
    pairwise_sim = mean([cos_sim(e1, e2) for e1, e2 in combinations(embeddings, 2)])
    return 1 - pairwise_sim  # diversity score
```

**(B) One-step entropy prediction** (moderate):
```python
def future_value_entropy(model, h_prime, K_future=3, S=10):
    # What's the best info gain available from h_prime?
    future_steps = sample_reasoning_steps(model, h_prime, K_future)
    best_gain = max(info_gain(model, h_prime, a, S) for a in future_steps)
    return best_gain
```

**(C) LLM self-assessment** (cheapest but least reliable):
```python
def future_value_llm(model, h_prime):
    prompt = f"""Given this reasoning state, estimate:
    1. How many plausible distinct answers remain? (1-100)
    2. How many productive next steps are available? (0-10)
    Respond with just two numbers."""
    response = model.generate(prompt + h_prime)
    n_answers, n_steps = parse(response)
    return log(n_answers) * (n_steps > 0)  # entropy-like × feasibility
```

### 3.2 Corridor-Specific Evaluation

**Construct a "corridor test set"** from MATH to specifically test the planning-ahead advantage:

1. From MATH-500, identify problems where the reference solution contains intermediate reformulation steps (e.g., variable substitution, applying an identity, converting between representations)
2. These reformulations are "corridor steps" — they don't directly constrain the answer but enable subsequent progress
3. Manual annotation of ~50 such problems (or LLM-assisted identification)

**Metric:** On corridor problems specifically, does planning-aware search outperform greedy? This is the core claim about Sun et al.'s framework.

### 3.3 Discount Factor Sweep

Test γ ∈ {0, 0.3, 0.5, 0.7, 0.9}:
- γ=0 reduces to greedy (Phase 1)
- γ too high → overvalues speculative future gains
- Expect optimal γ ∈ [0.3, 0.7] based on typical reasoning chain lengths

**Deliverable:** Comparison of greedy vs. planning-aware, overall and on corridor subset. Analysis of which problems benefit from planning ahead.

---

## 4. Phase 3: Full Evaluation & Paper-Ready Results (Week 6–8)

### Goal
Produce the complete evaluation for a paper submission.

### 4.1 Extended Benchmarks

| Benchmark | Size | Why |
|---|---|---|
| MATH-500 | 500 | Primary. Full difficulty range. |
| AIME 2024 | 30 | Hard competition math. High corridor density. |
| AMC 2023 | 25 | Medium difficulty. |
| GSM8K-Hard | 200 subset | Tests whether the method degrades on easy problems where search isn't needed. |

### 4.2 Model Sweep

Test across model families to show generality:

| Model | Size | Access |
|---|---|---|
| Qwen2.5-Math-7B | 7B | Fireworks API |
| Qwen2.5-Math-72B | 72B | Fireworks API |
| DeepSeek-R1-distill-7B | 7B | Fireworks API (if available) |
| Kimi K2.5 | — | Fireworks API |

**Key question:** Does curiosity search help more with weaker models (where the base policy is worse and search adds more value) or stronger models (where calibration is better)?

### 4.3 Compute-Matched Comparisons

The most important table in the paper:

| Method | MATH-500 Acc | AIME Acc | Forward passes/problem | Requires verifier? |
|---|---|---|---|---|
| Chain-of-Thought (1 sample) | baseline | baseline | ~1× | No |
| Best-of-N (N=50) | — | — | ~50× | No |
| Self-Consistency (N=50) | — | — | ~50× | No |
| ToT + self-eval | — | — | matched | No |
| MCTS + PRM (if available) | — | — | matched | **Yes** |
| **Greedy curiosity (ours)** | — | — | matched | No |
| **Planning-aware curiosity (ours)** | — | — | matched | No |
| Oracle expansion | — | — | matched | Ground truth |

### 4.4 Analysis Sections for Paper

1. **Scaling curves**: Accuracy vs. compute budget for each method
2. **Corridor analysis**: Case studies of problems where planning-aware curiosity succeeds and greedy fails
3. **Entropy trajectory visualization**: Show how answer entropy decreases over search expansions for representative problems
4. **Calibration analysis**: Report calibration metrics, show the method's accuracy degrades with calibration quality
5. **Proxy comparison**: Exact entropy vs. embedding proxy vs. token entropy
6. **Failure modes**: Where does curiosity search fail? (Hypothesis: problems where the answer space is continuous or poorly discretized)

---

## 5. Phase 4 (Optional): Multi-Hop QA Extension (Week 8–12)

### Goal
Show the framework generalizes beyond math to a domain where "corridors" are even more prominent: multi-hop reasoning requiring intermediate entity retrieval.

### 5.1 Setup

**Datasets:** HotpotQA (distractor setting), MuSiQue

**Mapping:**
- Action a = choose which reasoning hop to take next (which sub-question to answer)
- Observation o = retrieved passage or intermediate answer
- H(Answer|h) = entropy over final answer given partial reasoning
- Corridor = bridge entity retrieval (uninformative in isolation, necessary for final answer)

### 5.2 Adaptation Required

- Answer space is open-ended text → need semantic clustering for entropy
- Retrieval steps are non-deterministic (unlike math) → N > 1 matters
- May need retriever integration (contriever or similar)

**This phase is contingent on Phase 3 showing positive results on math.**

---

## 6. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Entropy estimates too noisy** | 30% | Fatal | Phase 0 tests this first. Semantic entropy as backup. If both fail, stop. |
| **Greedy curiosity ≈ random** | 20% | High | Analyze whether the problem is noise (more samples) or fundamental (information gain doesn't correlate with useful steps). |
| **Planning-ahead adds nothing over greedy** | 40% | Medium | Still publishable with greedy results + theoretical contribution from Sun et al. mapping. |
| **Compute budget too high for fair comparison** | 25% | Medium | Embedding proxy reduces cost ~20×. Lead with proxy results if exact is too expensive. |
| **ToT/MCTS baselines too strong** | 20% | Medium | Contribution is verifier-free search. Even matching verifier-based methods without a verifier is a result. |
| **Only works on one model** | 15% | Medium | Model sweep in Phase 3 catches this early. |
| **Calibration degrades at hard problems** | 40% | High | Hard problems are where search matters most. If calibration fails precisely where it's needed, the approach has limited value. Diagnose in Phase 0/1 by difficulty stratification. |

---

## 7. Compute Budget

### API Costs (Fireworks)

| Phase | Forward passes | Est. tokens | Est. cost (7B model) |
|---|---|---|---|
| Phase 0: Calibration | 50K | 25M | ~$5 |
| Phase 1: Greedy search (500 problems × 250 passes) | 125K | 60M | ~$12 |
| Phase 1: Baselines (matched) | 125K | 60M | ~$12 |
| Phase 1: Embedding proxy ablation | 25K | 12M | ~$3 |
| Phase 2: Planning-aware (500 × 400 passes) | 200K | 100M | ~$20 |
| Phase 3: Model sweep (4 models × 200K) | 800K | 400M | ~$80–200 |
| **Total through Phase 3** | **~1.3M** | **~650M** | **~$130–250** |

72B model runs are ~10× more expensive per token. Budget the 72B sweep separately.

### GPU Compute

- Sentence-transformer encoding: Trivial on A100 (~1000 encodings/sec)
- If running local models instead of API: Qwen2.5-Math-7B fits on single A100 with vLLM, ~100 tokens/sec. Phase 1 would take ~7 hours of continuous inference.

---

## 8. Timeline

```
Week 1-2:  Phase 0 — Calibration validation
           GO/NO-GO DECISION
Week 2-4:  Phase 1 — Greedy curiosity search + baselines + embedding proxy
           INTERIM RESULTS: Does info gain beat random expansion?
Week 4-6:  Phase 2 — Planning-aware search + corridor analysis
           KEY QUESTION: Does planning ahead help on corridor problems?
Week 6-8:  Phase 3 — Full eval, model sweep, paper writing
           DELIVERABLE: Paper draft
Week 8-12: Phase 4 (optional) — Multi-hop QA extension
           DELIVERABLE: Extended paper / second contribution
```

---

## 9. Paper Outline

**Title:** "Planning to Be Surprised: Curiosity-Guided Inference-Time Search for LLM Reasoning"
(Intentional echo of Sun et al. 2011 title)

1. **Introduction**: Inference-time search is reward-dependent; we propose information-gain-guided search grounded in optimal Bayesian exploration theory.
2. **Background**: Sun et al. 2011 framework, curiosity Q-values, key properties (additive only in expectation, corridor problem).
3. **Method**: Formal mapping to reasoning, greedy and planning-aware algorithms, computational approximations (semantic entropy, embedding proxy).
4. **Experimental Setup**: Benchmarks, baselines, compute matching.
5. **Results**: Main comparison, scaling curves, corridor analysis, proxy ablation, model sweep.
6. **Analysis**: When and why curiosity search helps; failure modes; calibration sensitivity.
7. **Related Work**: (already written — Section 10 of the spec doc)
8. **Conclusion**: Verifier-free search via information gain is competitive; planning-ahead matters for corridor problems; Schmidhuber's 1991 framework transfers to modern LLM reasoning.

**Target venues:** EMNLP 2026, NeurIPS 2026, or ICLR 2027 depending on timing.

---

## 10. Decision Points

### After Phase 0:
- **Entropy reliable** → proceed as planned
- **Entropy noisy but salvageable** → switch to semantic entropy + increase S → budget increases ~2×
- **Entropy fundamentally unreliable** → pivot to embedding-only proxy (weaker theoretical story but still publishable as empirical contribution) OR pivot back to training-time curiosity (your existing plan v4)

### After Phase 1:
- **Greedy curiosity > ToT** → strong result, proceed to planning-aware as bonus
- **Greedy curiosity ≈ ToT** → planning-aware is needed to differentiate, Phase 2 becomes critical
- **Greedy curiosity < ToT** → diagnose. If close, push to Phase 2. If far, check if embedding proxy does better (possible if exact entropy is noisy but directionally correct)

### After Phase 2:
- **Planning-aware > greedy on corridor problems** → paper writes itself
- **Planning-aware ≈ greedy everywhere** → still publishable (negative result about corridors in LLM reasoning is interesting) but weaker paper

### After Phase 3:
- **Works across models** → strong generality claim
- **Only works on one model family** → qualify claims, still publishable but narrower

---

## 11. Relationship to Existing Plan (v4)

This is a **parallel track**, not a replacement for the agent-based ETE work.

| Dimension | Plan v4 (Agent ETE) | This plan (Inference-Time Search) |
|---|---|---|
| Setting | Agent in text games (ScienceWorld, Jericho) | Standalone reasoning (MATH, AIME) |
| When curiosity acts | During episode, driving action selection | During inference, driving search tree expansion |
| Learning signal | Observation surprise (cosine sim) | Information gain (answer entropy reduction) |
| Training | GRPO with intrinsic reward | No training — zero-shot search |
| Schmidhuber connection | 1990/1991 curiosity signals | 2011 optimal Bayesian exploration |
| Compute | A100 for GRPO training runs | API calls for inference |
| Timeline overlap | Can run simultaneously | API-only, doesn't compete for GPU |

**The two are complementary and could be combined in a single paper** with a framing like "Schmidhuber's curiosity framework for LLM agents: from training to inference." The agent work shows curiosity helps *learn better policies*; the inference work shows curiosity helps *search better at test time*. Together they cover both sides of the problem.

Alternatively, they're two separate papers with different contribution types (empirical RL paper vs. theory-grounded inference paper).
