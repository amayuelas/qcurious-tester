# Curiosity-Driven Planning for LLM Test Generation

Code and benchmarks for the paper *"Curiosity-Driven Planning for LLM Test Generation"* (COLM 2026).

We formalize LLM-based test generation as Bayesian exploration of an unknown environment, where the coverage map serves as the posterior and an LLM-estimated Q-value selects among diverse candidate plans. Our method, **CovQValue**, achieves 40--77% more branch coverage than greedy baselines across 233 targets, 19 repositories, and 3 LLMs.

## Setup

```bash
# Install with uv (preferred)
uv pip install -e .

# Or pip
pip install -e .
```

Requires Python >= 3.10. Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

## Benchmarks

- **RepoExploreBench** (ours): 93 modules from 9 popular Python packages. Requires the `curiositybench:latest` Docker image.
- **TestGenEval Lite**: 140 files from 10 repos. Uses SWE-bench Docker images.

## Running Experiments

```bash
# RepoExploreBench (93 targets, 9 repos)
python run_repo_explore_bench.py [--max-targets N] [--repos X Y Z] [--strategies S...] [--seeds N...]

# TestGenEval Lite (140 targets, 10 repos)
python run_testgeneval.py [--repos R...] [--max-examples N] [--exec-budget B] [--strategies S...]

# Ablation studies
python run_ablations.py              # gamma, K, S sweeps
python run_ablation_diversity.py     # diversity decomposition
python run_ablation_s_matched.py     # S with matched rounds

# Quick smoke test
python run_repo_explore_bench.py --max-targets 1 --exec-budget 6
```

## Generating Figures

```bash
python plots/paper_figures.py        # Main paper figures
python plots/paper_ablations.py      # Ablation figures and tables
python plots/paper_case_studies.py   # Corridor navigation case studies
```

Output goes to `plots/paper/`.

## Project Structure

```
curiosity_explorer/
  explorer/
    coverage_exploration.py   # Core: CoverageMap, strategies, plan generation
    q_values.py               # Q-value scoring (immediate gain + future reachability)
    diverse_gen.py            # Diverse candidate generation
    entropy_utils.py          # Entropy computation utilities
  runner/
    docker_coverage.py        # Docker-based test execution and coverage measurement
  benchmarks/
    repo_explore_bench.py     # RepoExploreBench: 93 targets, 9 repos
    testgeneval_config.py     # TestGenEval Lite: per-repo Docker configs
  llm.py                      # Multi-model LLM client (Gemini, GPT, Mistral)

run_repo_explore_bench.py     # RepoExploreBench experiment runner
run_testgeneval.py            # TestGenEval experiment runner
run_ablations.py              # Ablation studies runner
config.py                     # API keys, model config, experiment parameters
plots/                        # Figure generation scripts
results/                      # Experiment outputs (JSON)
```

## Strategies

| Strategy | Coverage Feedback | Plan Selection | Planning |
|---|---|---|---|
| Random | No | Random | No |
| Greedy | No | LLM picks "best" | No |
| CovGreedy | Yes | Random from K | No |
| **CovQValue** | Yes | Q-value scoring | Yes |

## Citation

```bibtex
@article{amayuelas2026planning,
  title={Planning to Explore: Curiosity-Driven Planning for LLM Test Generation},
  author={Alfonso Amayuelas and Firas Laakom and Piotr Piękos and Wenyi Wang and Yifan Xu and Yuhui Wang and Jürgen Schmidhuber and William Wang},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026},
  journal={arXiv preprint arXiv:2604.05159},
}
```
