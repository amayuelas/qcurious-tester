#!/usr/bin/env python3
"""Download experiment results from Hugging Face Hub.

Usage:
    python scripts/download_results.py [--repo-id USER/DATASET_NAME]

Downloads all result JSONs needed to reproduce paper figures into results/.
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

REPO_ID_DEFAULT = "amayuelas/qcurious-tester"

RESULT_FILES = [
    "repo_explore_bench/full_run_gemini.json",
    "repo_explore_bench/full_run_gpt54mini.json",
    "repo_explore_bench/full_run_mistral.json",
    "testgeneval/full_run_gemini.json",
    "testgeneval/full_run_gpt54mini.json",
    "testgeneval/full_run_mistral.json",
    "repo_explore_bench/exec_selection_gemini.json",
    "repo_explore_bench/exec_selection_gpt54mini.json",
    "repo_explore_bench/exec_selection_mistral.json",
    "testgeneval/exec_selection_gemini.json",
    "testgeneval/exec_selection_gpt54mini.json",
    "testgeneval/exec_selection_mistral.json",
    "ablations/ablation_exec_budget.json",
    "ablations/ablation_S_matched.json",
    "ablations/ablation_gamma.json",
    "ablations/ablation_K_plans.json",
    "ablations/ablation_diversity.json",
    "ablations/ablation_S_plan_length.json",
]


def main():
    parser = argparse.ArgumentParser(description="Download results from HF Hub")
    parser.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    args = parser.parse_args()

    downloaded = 0
    for rel_path in RESULT_FILES:
        dest = RESULTS_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {rel_path}...")
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=rel_path,
            repo_type="dataset",
            local_dir=str(RESULTS_DIR),
        )
        downloaded += 1

    print(f"\nDone: {downloaded} files downloaded to {RESULTS_DIR}/")
    print("Run `python plots/paper_figures.py` to regenerate figures.")


if __name__ == "__main__":
    main()
