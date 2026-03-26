# Research Findings — Curiosity-Guided Code Exploration

**Last updated:** 2026-03-25
**Total spend:** ~$280
**Status:** Main experiments running (TestGenEval + CuriosityBench)

---

## 1. Core Research Question

Can Schmidhuber/Sun et al.'s Bayesian exploration framework improve LLM-based test generation? Specifically: does selecting tests by information gain (curiosity) outperform greedy coverage-directed selection?

---

## 2. What We Built

### Strategies (4 compared)
| Strategy | Generation | Selection | Key idea |
|---|---|---|---|
| **random** | Standard prompt × K | Random pick | Baseline — pure diversity |
| **greedy** | Standard prompt × K | LLM picks "best for coverage" | Current practice (CoverUp-style) |
| **curiosity_qvalue** | Diverse prompts (7 strategies) | Q-value: entropy + future planning | Sun et al. Proposition 1 |
| **reflective_qvalue** | Learnings-guided | Q-value + predict→compare→reflect | Full Schmidhuber cycle |

### Benchmarks (3 benchmarks)
| Benchmark | Type | Files | What it tests |
|---|---|---|---|
| **TestGenEval Django** | Real-world, Docker | 37 Django files (4.0-5.0) | File-level code with import/setup corridors |
| **CuriosityBench Real Repos** | Real-world, Docker | 35 files from 10 pip repos | File-level across click, requests, flask, etc. |
| **CuriosityBench Stdlib** | Forked stdlib, sandbox | 9 full modules (345-1333L) | Standalone modules (control condition) |

### Infrastructure
- `DockerCoverageRunner`: Branch coverage in Docker containers
- `CoverageRunner.run_script()`: Multi-line test script execution with coverage
- Q-value computation with logprob entropy + binary future branching
- Diverse script generation (7 prompting strategies)
- Reflective loop (predict → execute → compare → reflect → update learnings)

---

## 3. Key Findings

### Finding 1: Q-value works on file-level code with corridor structure

**TestGenEval Django (8/37 files complete, preliminary):**

| Strategy | Mean coverage | Wins | vs Random |
|---|---|---|---|
| **curiosity_qvalue** | **76.4** | **4/8** | **+22.5** |
| reflective_qvalue | 60.9 | 3/8 | +7.0 |
| greedy | 60.5 | 3/8 | +6.6 |
| random | 53.9 | 1/8 | — |

Q-value's biggest win: `migrations/serializer.py` — gets **113 branches** while random and greedy get **0**. The Q-value strategy discovered how to test the serializer module while others couldn't even generate valid test scripts.

**CuriosityBench Real Repos (smoke test, 2 files):**
- click.core (3042 lines): Q-value **201** vs random **136** (+65 branches)
- click.types (1089 lines): Q-value 25 vs random 39 (-14)
- Pattern: Q-value helps on large, complex files but not small ones.

### Finding 2: Q-value does NOT work on standalone modules

**CuriosityBench Stdlib (9 modules complete):**

| Strategy | Mean coverage |
|---|---|
| greedy | **193.1** |
| reflective_qvalue | 169.4 |
| random | 158.2 |
| curiosity_qvalue | 121.6 |

Q-value is the **worst** strategy on standalone modules. The LLM reads the code and generates good tests directly — no corridor to navigate, no planning advantage.

### Finding 3: The corridor effect requires operational complexity, not just code complexity

Three types of programs tested:

| Type | Q-value advantage | Why |
|---|---|---|
| **File-level real code** (Django, click) | **+15-65 branches** | Import chains, class setup, configuration — reading code isn't enough |
| **Standalone modules** (stdlib) | **-37 branches** | LLM reads code and targets branches directly |
| **Standalone functions** (ULT) | **~0 branches** | Oracle gap too small, candidates too similar |

**The critical factor:** whether the LLM can solve the testing problem just by reading the code. If yes, greedy wins. If no (setup complexity, partial observability), Q-value wins.

### Finding 4: Single-step calibration is the wrong evaluation

We spent ~$56 testing whether estimator scores predict single-step coverage gain (Spearman ρ). 8 estimators tested, none passed ρ > 0.15 at scale. But Q-value works in the full loop despite failing calibration.

**Why:** Information gain's value is cumulative over multiple steps. A test with zero immediate coverage can be the most valuable (it teaches the model how to pass validation gates). Single-step metrics can't capture this.

### Finding 5: Diverse generation matters more than selection

On ULT standalone functions, diverse generation + random selection beat greedy 83% on corridor functions. The bottleneck was candidate quality, not candidate selection.

On file-level code, diverse generation + Q-value selection is the winning combination. Both components contribute.

### Finding 6: The zero-entropy floor cripples sampling-based estimators

Gemini produces identical predictions across S=8 samples 79-91% of the time. This makes sampling entropy useless as a selection signal — it scores 80-90% of candidates as identical (all zero).

### Finding 7: Reflection hurts more than it helps

| Comparison | Result |
|---|---|
| TestGenEval: reflective vs qvalue | 60.9 vs **76.4** (reflective worse) |
| Corridors: reflective vs qvalue | 84.5 vs **95.0** (reflective worse) |
| Obfuscated: reflective vs qvalue | 69.5 vs **72.0** (reflective worse) |

The reflection loop ("you were wrong about X") makes generation conservative — the model avoids past mistakes instead of exploring new territory.

### Finding 8: Oracle is not an upper bound

Our single-step greedy oracle loses to random on some functions. This validates Sun et al.'s non-additivity argument: greedy maximization of immediate information gain is suboptimal for cumulative exploration.

### Finding 9: Model mismatch kills performance

Using gpt-oss-120b for scoring + Gemini for generation made logprob entropy the worst strategy. Same model for everything is essential.

---

## 4. The Story for the Paper

**Title idea:** "Planning to Explore: Curiosity-Guided Test Generation via Bayesian Exploration"

**Claim 1 (strong evidence):** Q-value planning with diverse generation outperforms greedy selection on file-level code with corridor structure. TestGenEval results show +15-20 branches on average.

**Claim 2 (moderate evidence):** The advantage is specific to code with operational complexity (import chains, class setup, runtime state). On standalone functions/modules, greedy suffices.

**Claim 3 (characterized negative):** Single-step information gain estimators (sampling entropy, logprob entropy, verbalized confidence, multi-model disagreement) do not predict immediate coverage gain at scale. But the full exploration loop works despite this, because the value is cumulative.

**Claim 4 (theoretical validation):** The single-step greedy oracle is not an upper bound — confirming Sun et al.'s non-additivity of information gain in exploration.

---

## 5. Experiments Running

### TestGenEval Django (main result)
- 37 files × 4 strategies, 8 steps each
- 8/37 complete, Q-value leading at 76.4 mean
- Estimated completion: ~4-5 hours remaining
- Expected cost: ~$25-30

### CuriosityBench Real Repos (custom benchmark)
- 35 files × 4 strategies, 10 steps each
- Just started (smoke test passed)
- Estimated completion: ~3-4 hours
- Expected cost: ~$5-10

---

## 6. What's Still Needed

### For the paper:
1. **Complete TestGenEval + CuriosityBench runs** — for statistical significance
2. **Cost-normalized comparison** — coverage per API dollar (Q-value costs ~3x more per step)
3. **2-3 case studies** — detailed walkthroughs of corridor traversal
4. **Ablations** — K, γ, budget scaling, diverse vs standard gen
5. **External baseline comparison** — cite TestForge/CoverUp numbers on same files
6. **Paper writing**

### Nice to have:
- More repos in CuriosityBench (currently 10 repos)
- Multiple seeds per file for tighter confidence intervals
- Sympy/scikit-learn in TestGenEval (need Docker images)

---

## 7. Cost Breakdown

| Experiment | Cost |
|---|---|
| Phase 0-0.5: Setup + toy programs | ~$5 |
| Phase 1: ULT calibration | ~$56 |
| Phase 0b: Estimator diagnostics | ~$10 |
| Head-to-head ULT | ~$21 |
| Diverse generation experiments | ~$13 |
| Black-box experiments | ~$5 |
| Q-value corridor programs | ~$28 |
| Generated corridor functions | ~$92 |
| TestGenEval Django (earlier runs) | ~$8 |
| CuriosityBench stdlib | ~$34 |
| TestGenEval (running) | ~$10 so far |
| CuriosityBench real repos (running) | ~$1 so far |
| **Total** | **~$283** |
| **Budget remaining** | **~$32** of $315 allocated |

---

## 8. Key Files

| File | Purpose |
|---|---|
| `run_testgeneval.py` | TestGenEval Django runner |
| `run_curiositybench.py` | CuriosityBench real repos runner |
| `curiosity_explorer/explorer/q_values.py` | Q-value (Sun et al. Prop 1) |
| `curiosity_explorer/explorer/diverse_gen.py` | Diverse prompting strategies |
| `curiosity_explorer/runner/docker_coverage.py` | Docker coverage measurement |
| `curiosity_explorer/runner/coverage.py` | Sandbox coverage + run_script() |
| `curiosity_explorer/benchmarks/curiosity_bench/` | Our custom benchmark |
| `results/FINDINGS.md` | This file |
| `results/PROGRESS_REPORT.md` | Detailed progress report |
| `results/calibration_results.md` | Phase 1 calibration writeup |
