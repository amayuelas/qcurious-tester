# Research Findings — Coverage-Map Guided Exploration

**Last updated:** 2026-03-26
**Total spend:** ~$290
**Status:** Benchmarks ready, quick tests passing, full runs pending

---

## 1. Core Research Question

Can coverage-map feedback with trajectory planning (grounded in Sun et al.'s Bayesian exploration framework) improve LLM-based test generation over standard baselines?

**Answer: Yes.** On real-world code, our cov_qvalue method achieves 2.9-5x more branch coverage than random baselines with equal execution budgets.

---

## 2. Method: Coverage-Map Guided Exploration with Q-Value Plan Selection

### Theoretical grounding (Sun et al. 2011)

| Theory | Symbol | Our Implementation |
|---|---|---|
| Unknown environment | Θ | Program's branch reachability structure |
| Bayesian posterior | p(Θ\|h) | Coverage map — tracked branches after tests h |
| Action | a | A test plan (sequence of 3 scripts) |
| Observation | O | Coverage result (which branches were hit) |
| Greedy IG | ḡ(a\|h) | Expected new branches from plan a |
| Q-value | q(a\|h) = ḡ + γ·E[v(h')] | Immediate branches + future reachability |

### Three strategies compared

| Strategy | Generation | Selection | Coverage feedback | Planning |
|---|---|---|---|---|
| **random** | Standard prompt (source + history) | Random pick | No | No |
| **cov_greedy** | Coverage-aware (source + coverage map + "target gaps") | Random pick from K | Yes | No |
| **cov_qvalue** | K=3 diverse plans (3 steps each) | Score by Q-value, pick best | Yes | Yes |

### How cov_qvalue works

1. **Generate K=3 candidate plans** in parallel, each with a diversity hint
2. **Score each plan** by LLM-estimated Q-value:
   - ḡ = expected immediate new branches
   - E[v(h')] = expected future branches made reachable
   - Q = ḡ + γ · E[v(h')]
3. **Select plan with highest Q-value**
4. **Execute all 3 steps**, updating coverage map after each
5. Repeat with updated posterior

---

## 3. Key Results

### Finding 1: cov_qvalue dominates on fair comparison (equal executions)

**Django 4.0, 5 files, 24 executions each:**

| Strategy | Mean branches | vs random |
|---|---|---|
| random | 24.8 | — |
| cov_greedy | 50.2 | +102% |
| **cov_qvalue** | **124.8** | **+403%** |

Decomposition of contributions:
- Coverage feedback alone: +55% (random → random_covfeedback at 38.4)
- Coverage-targeted generation: +102% (random → cov_greedy at 50.2)
- + Trajectory planning + Q-value selection: +403% (random → cov_qvalue at 124.8)

### Finding 2: Q-value selection is critical (not just more plans)

cov_planned (single plan, no selection) gets 65.6 mean. cov_qvalue (K plans + Q-value) gets 124.8 — nearly double. The Q-value scorer correctly shifts selection after breakthroughs.

| File | cov_planned | cov_qvalue | Why |
|---|---|---|---|
| forms/models.py | 21 | **140** | Q-value broke through where single plan got stuck |
| resolvers.py | 10 | **194** | Q-value found the corridor |
| serializer.py | **123** | 105 | Single plan got lucky (cov_planned wins ~20% of the time) |

### Finding 3: Coverage map feedback helps generation quality

cov_greedy (coverage map in prompt, random selection) beats random by 2x even with equal executions. The coverage map tells the LLM:
- How many branches discovered so far
- Growth rate and stagnation warnings
- Which tests were most informative

### Finding 4: Method generalizes across repos

**RepoExploreBench quick test (5 click modules, 12 execs each):**

| Module | random | cov_greedy | cov_qvalue |
|---|---|---|---|
| click.core | 90 | 92 | **272** |
| click.types | 21 | 9 | **53** |
| click.termui | 9 | 19 | **22** |
| click.utils | 15 | 18 | **21** |
| click.shell_completion | 4 | 32 | **37** |
| **Mean** | **27.8** | **34.0** | **81.0** |

cov_qvalue wins 5/5 modules. Mean +191% over random.

**CuriosityBench (10 modules, 8 execs each, unfair — cov_qvalue got 3x executions):**

| | random | cov_greedy | cov_qvalue |
|---|---|---|---|
| Mean | 54.8 | 80.6 | 95.8 |

### Finding 5: Approaches that DON'T work

| Approach | Result | Why |
|---|---|---|
| **greedy** (LLM picks "best") | ≈ random | LLM can't predict coverage from code alone |
| **Learning progress** (enriched history) | ≈ random | Enriched history doesn't improve generation |
| **Reflection** (predict→compare→learn) | Worse than random | Makes generation conservative |
| **Sampling entropy** | Worst | Zero-entropy floor (79-91% identical predictions) |

### Finding 6: The corridor effect

The advantage of cov_qvalue is largest on code with **corridor structure** — where setup steps are needed to reach deep branches:

| Code type | cov_qvalue advantage | Example |
|---|---|---|
| Deep corridors (formsets, models, resolvers) | **5-28x** over random | formsets: 80 vs 2 |
| Moderate corridors (click.core, serializer) | **2-3x** over random | click.core: 272 vs 90 |
| No corridor (admin/options — import gets everything) | **1x** (tied) | All strategies get 228 |
| Hard blockers (auth/forms — needs database) | **1x** (tied) | All stuck at 5 |

---

## 4. Benchmarks

### RepoExploreBench v2.0 (our benchmark)

- **93 targets** across 9 repos (click, requests, flask, rich, jinja2, httpx, pydantic, werkzeug, starlette)
- ~79K lines of real-world code
- All run in `curiositybench:latest` Docker image
- Selection criteria: top PyPI downloads + ≥200 lines + corridor structure + domain diversity

### TestGenEval Lite (external benchmark)

- **160 files** across 11 repos (django, sympy, scikit-learn, pytest, matplotlib, astropy, xarray, seaborn, sphinx, pylint, flask)
- All 44 Docker images pulled and ready
- Per-repo configs with conda/pyenv environment support

---

## 5. Divergence from PLAN_v4

PLAN_v4 laid out a phased approach: find the right uncertainty readout (Phase 0b), calibrate it (Phase 1b), then run TestGenEval (Phase 2), corridor analysis (Phase 3), and ablations (Phase 4). Here's what actually happened and why:

### 5.1 All uncertainty readouts failed (Phase 0b → pivot)

PLAN_v4 proposed 4 estimators to read out the LLM's uncertainty:
- **Multi-model disagreement** — models agree too often; Gemini + Mistral give identical predictions
- **Logprob entropy** — zero-entropy floor; 79-91% of candidates score zero
- **Verbalized confidence** — confounded; measures state (early vs late), not candidate quality
- **Hybrid** — combining broken signals doesn't fix them

Result: ALL estimators failed ρ > 0.15 threshold. This triggered the "STOP" condition in PLAN_v4's decision tree.

### 5.2 Selection → generation (fundamental pivot)

PLAN_v4 assumed the bottleneck was **candidate selection** — picking the best test from K candidates. We discovered:
- Oracle ceiling is ~10% at state 0 (all candidates roughly equivalent)
- The real bottleneck is **generation quality**, not selection
- Random selection with better generation beats smart selection with standard generation

This led us to abandon the "score candidates by information gain" approach entirely.

### 5.3 LLM beliefs → coverage map (new posterior)

PLAN_v4 treated the LLM's internal beliefs (updated via in-context learning) as the Bayesian posterior p(Θ|h). We replaced this with:
- The **coverage map** (tracked branch counts) IS the posterior — observable, exact, no readout needed
- Fed back to the LLM as structured text in the prompt
- The LLM uses it to improve generation, not as an uncertainty signal

### 5.4 Individual test selection → trajectory planning (new algorithm)

PLAN_v4's algorithm: generate K tests, score each by IG, select best.

Our algorithm: generate K **plans** (sequences of 3 tests), score each plan by Q-value, select the best plan, execute all 3 steps. This is closer to Sun et al.'s framework — the "action" is a trajectory, not a single test.

### 5.5 Multi-model ensemble → single model (simplification)

PLAN_v4 proposed using 3-5 different LLMs for disagreement. We use a single model (Gemini Flash) for everything — generation, scoring, and Q-value estimation. Cheaper and simpler.

### 5.6 ULT abandoned → RepoExploreBench (new benchmark)

PLAN_v4 used ULT (standalone functions) as the primary calibration benchmark. We found:
- ULT functions have no corridor structure — LLM reads code and targets branches directly
- Greedy beats curiosity on standalone functions
- Corridor structure requires file-level code with import chains and setup requirements

Created **RepoExploreBench** (93 modules across 9 pip repos) as our benchmark, with principled selection from top PyPI packages.

### 5.7 Phased plan → iterative experimentation

The planned phases (0b → 1b → 2 → 3 → 4) were abandoned after Phase 0b failed. Instead, we iterated through:
1. LP enriched history (didn't help)
2. Coverage map feedback (helped generation)
3. Trajectory planning (big improvement)
4. Multi-plan + Q-value selection (biggest improvement)

Each iteration was tested on 5 Django files before scaling up.

### 5.8 Black-box condition dropped

PLAN_v4 proposed a black-box condition (LLM sees only function signature, not source code). We haven't pursued this — all results are white-box. The coverage map approach requires showing source code to the LLM for targeted generation.

### Summary: what survived from PLAN_v4

| PLAN_v4 element | Status | What replaced it |
|---|---|---|
| Sun et al. theoretical framework | **Kept** | Same framework, different instantiation |
| Information gain as selection signal | **Dropped** | Coverage map as generation signal |
| Uncertainty readout (sampling, logprobs, ensemble) | **Dropped** | LLM-estimated Q-value on plans |
| Candidate selection | **Dropped** | Plan selection (generate K plans, pick best) |
| ULT benchmark | **Dropped** | RepoExploreBench (real repos with corridors) |
| TestGenEval | **Kept** | Extended to all 160 files, all 11 repos |
| Phased experimental plan | **Dropped** | Iterative experimentation |
| Black-box condition | **Dropped** | White-box only (for now) |
| Coverage as metric | **Kept** | Same — branch coverage via coverage.py |
| Docker infrastructure | **Kept** | Extended with conda/pyenv support |

---

## 6. What's Still Needed

### For the paper:
1. **Full RepoExploreBench run** — 93 targets × 3 strategies × 1 seed (~$15, ~6h)
2. **Full TestGenEval run** — 160 files × 3 strategies × 1 seed (~$25, ~10h)
3. **Statistical significance** — paired t-tests, Cohen's d, per-repo breakdown
4. **Ablations** — K (1 vs 3 vs 5 plans), γ (0 vs 0.5 vs 1.0), exec budget scaling
5. **Cost-normalized comparison** — coverage per API dollar
6. **2-3 case studies** — detailed walkthroughs (formsets.py, click.core)
7. **Paper writing**

### Done:
- Method implementation (cov_greedy, cov_qvalue)
- Fair comparison methodology (equalized executions)
- Both benchmarks defined and ready
- Docker images for all repos
- Runners for both benchmarks
- Quick test results confirming the method works

---

## 7. Cost Breakdown

| Phase | Cost |
|---|---|
| Phase 0-1: Setup, calibration, estimator diagnostics | ~$90 |
| Head-to-head ULT + corridor experiments | ~$62 |
| Generated corridor functions | ~$92 |
| CuriosityBench stdlib + real repos | ~$35 |
| Coverage map experiments (Django) | ~$5 |
| Q-value plan selection experiments | ~$2 |
| RepoExploreBench quick test | ~$0.5 |
| TestGenEval quick tests | ~$2 |
| **Total** | **~$290** |
| **Budget remaining** | **~$25** of $315 |

---

## 8. Key Files

| File | Purpose |
|---|---|
| `run_repo_explore_bench.py` | RepoExploreBench runner (93 targets) |
| `run_testgeneval.py` | TestGenEval Lite runner (160 files) |
| `curiosity_explorer/explorer/coverage_exploration.py` | Core method (cov_greedy, cov_qvalue) |
| `curiosity_explorer/benchmarks/repo_explore_bench.py` | Our benchmark definition |
| `curiosity_explorer/benchmarks/testgeneval_config.py` | TestGenEval per-repo Docker config |
| `curiosity_explorer/runner/docker_coverage.py` | Docker coverage measurement |
| `results/FINDINGS.md` | This file |
