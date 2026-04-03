---
license: mit
task_categories:
  - text-generation
tags:
  - code-generation
  - test-generation
  - exploration
  - coverage
pretty_name: Planning to Explore - Experiment Results
---

# Planning to Explore: Experiment Results

Raw experiment results for the paper **"Planning to Explore: Curiosity-Driven Planning for LLM Test Generation"**.

These JSON files contain all data needed to reproduce the figures and tables in the paper.

## Usage

```bash
# Clone the code repository
git clone https://github.com/amayuelas/qcurious-tester.git
cd qcurious-tester

# Download results
python scripts/download_results.py

# Regenerate figures
python plots/paper_figures.py
python plots/paper_ablations.py
```

## File Structure

```
repo_explore_bench/
  full_run_{gemini,gpt54mini,mistral}.json    # Main results (Table 1, Figures 2-5)
  exec_selection_{gemini,gpt54mini,mistral}.json  # Execution-based selection (Appendix)
testgeneval/
  full_run_{gemini,gpt54mini,mistral}.json    # Main results (Table 1, Figures 2-5)
  exec_selection_{gemini,gpt54mini,mistral}.json  # Execution-based selection (Appendix)
ablations/
  ablation_exec_budget.json                   # Figure 6a
  ablation_S_matched.json                     # Figure 6b
  ablation_gamma.json                         # Figure 6c (top)
  ablation_K_plans.json                       # Figure 6c (top)
  ablation_diversity.json                     # Figure 6c (bottom)
  ablation_S_plan_length.json                 # Figure 6b (extended)
```

## Benchmarks

- **RepoExploreBench** (93 targets, 9 repos): click, requests, flask, rich, jinja2, httpx, pydantic, werkzeug, starlette
- **TestGenEval Lite** (140 targets, 11 repos): from SWE-bench

## Models

- Gemini 3 Flash (Google)
- GPT-5.4 Mini (OpenAI)
- Mistral Large 3 (Mistral AI)

## Citation

```bibtex
@inproceedings{amayuelas2026planning,
  title={Planning to Explore: Curiosity-Driven Planning for LLM Test Generation},
  author={Amayuelas, Alfonso and Laakom, Firas and Pi\k{e}kos, Piotr and Wang, Wenyi and Xu, Yifan and Wang, Yuhui and Schmidhuber, J\"urgen and Wang, William},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```
