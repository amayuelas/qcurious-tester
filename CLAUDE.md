# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of curiosity-guided inference-time search for LLM reasoning, based on Sun et al.'s (2011) Bayesian exploration framework. The core question: does information-gain-driven tree search outperform reward-based baselines on code exploration tasks?

The theoretical mapping: coverage map = Bayesian posterior p(Θ|h), test plan = action, coverage result = observation, Q-value = immediate information gain + discounted future reachability.

## Setup & Installation

```bash
# Install with uv (preferred)
uv pip install -e .

# Or pip
pip install -e .
```

Requires Python ≥3.10. API keys are configured via `.env` (loaded by `config.py` through `python-dotenv`). The active LLM model is set via the `MODEL` env var (default: `gemini-3-flash-preview`).

## Running Experiments

```bash
# RepoExploreBench (93 targets, 9 repos) — requires curiositybench:latest Docker image
python run_repo_explore_bench.py [--max-targets N] [--repos X Y Z] [--strategies S...] [--seeds N...]

# TestGenEval Lite (160 files, 11 repos) — requires SWE-bench Docker images
python run_testgeneval.py [--repos R...] [--max-examples N] [--exec-budget B] [--strategies S...]

# Ablation studies
python run_ablations.py              # γ, K, S sweeps
python run_ablation_diversity.py     # diverse generation ablation
python run_ablation_s_matched.py     # S-parameter matching

# Quick smoke tests
python run_repo_explore_bench.py --max-targets 1 --exec-budget 6
python run_testgeneval.py --max-examples 2 --exec-budget 6
```

Experiment parameters (`BUDGET`, `K`, `S`) can be set via env vars or CLI args.

## Generating Figures

```bash
python plots/paper_figures.py       # Main paper figures
python plots/paper_ablations.py     # Ablation figures and tables
python plots/plot_results.py        # Generic result plotting
```

Output goes to `plots/*.png`. Figures are sized for 50% LaTeX textwidth.

## Architecture

### Core Package: `curiosity_explorer/`

- **`llm.py`** — Multi-model LLM client (OpenAI, Gemini, Mistral, Fireworks) via OpenAI-compatible API. Handles response caching (temperature=0 only), batch generation with ThreadPoolExecutor, logprob extraction, and per-model cost tracking.

- **`explorer/coverage_exploration.py`** — Core algorithm. `CoverageMap` tracks the Bayesian posterior (known branch reachability). Three strategies: `random` (baseline), `cov_greedy` (coverage feedback in prompt), `cov_qvalue` (generate K plans, score by Q-value, execute best).

- **`explorer/q_values.py`** — `compute_q_values()` scores candidate test plans: immediate IG via logprob entropy + discounted future value via 1-step lookahead simulation. Falls back to sampling entropy when logprobs are unavailable.

- **`explorer/entropy_utils.py`** — String-based and logprob-based entropy computation.

- **`explorer/diverse_gen.py`** — Diverse candidate generation for plan proposals.

- **`runner/docker_coverage.py`** — `DockerCoverageRunner` executes test scripts in Docker containers and measures cumulative branch coverage via `coverage.py`.

- **`benchmarks/repo_explore_bench.py`** — RepoExploreBench: 93 curated targets across click, requests, flask, rich, jinja2, httpx, pydantic, werkzeug, starlette.

- **`benchmarks/testgeneval_config.py`** — TestGenEval Lite: 160 files across 11 repos with per-repo Docker config (image, python path, setup code).

### Top-Level Runners

`run_repo_explore_bench.py`, `run_testgeneval.py`, `run_ablations.py` — orchestrate experiments, call into `curiosity_explorer/`, write results to `results/*.json`.

### Configuration

`config.py` — Central config: API endpoints/keys from `.env`, active model, experiment parameters (`BUDGET`, `K`, `S`), model pricing table. All LLM providers are accessed through OpenAI-compatible clients with different base URLs.

### Data Flow

```
config.py → runner scripts → llm.py (generate/score) + explorer/ (CoverageMap, Q-values)
    → runner/docker_coverage.py (execute, measure coverage) → results/*.json
    → plots/*.py → plots/*.png
```

## Key Research Concepts

- **Corridor structure**: Import chains and class setup sequences that gate access to deep branches. This is where curiosity-guided exploration has the largest advantage (5–28× over random).
- **Three strategies compared**: `random` (generate + random select), `cov_greedy` (coverage feedback in prompt), `cov_qvalue` (K plans scored by Q-value with trajectory planning).
- **Q-value decomposition**: q(a|h) = ḡ(a|h) + γ·E[v(h')] where ḡ is immediate info gain and v(h') is future reachability value.
