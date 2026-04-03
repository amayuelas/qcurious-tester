#!/usr/bin/env python3
"""Upload experiment result JSONs to Hugging Face Hub.

Usage:
    python scripts/upload_results_to_hf.py [--repo-id USER/DATASET_NAME]

Requires: huggingface-cli login (or HF_TOKEN env var)
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Files needed to reproduce all paper figures and tables
RESULT_FILES = [
    # Main results (paper_figures.py)
    "repo_explore_bench/full_run_gemini.json",
    "repo_explore_bench/full_run_gpt54mini.json",
    "repo_explore_bench/full_run_mistral.json",
    "testgeneval/full_run_gemini.json",
    "testgeneval/full_run_gpt54mini.json",
    "testgeneval/full_run_mistral.json",
    # Execution-based selection (appendix)
    "repo_explore_bench/exec_selection_gemini.json",
    "repo_explore_bench/exec_selection_gpt54mini.json",
    "repo_explore_bench/exec_selection_mistral.json",
    "testgeneval/exec_selection_gemini.json",
    "testgeneval/exec_selection_gpt54mini.json",
    "testgeneval/exec_selection_mistral.json",
    # Ablations (paper_ablations.py)
    "ablations/ablation_exec_budget.json",
    "ablations/ablation_S_matched.json",
    "ablations/ablation_gamma.json",
    "ablations/ablation_K_plans.json",
    "ablations/ablation_diversity.json",
    "ablations/ablation_S_plan_length.json",
]


def main():
    parser = argparse.ArgumentParser(description="Upload results to HF Hub")
    parser.add_argument(
        "--repo-id",
        default="amayuelas/qcurious-tester",
        help="HF dataset repo ID (default: amayuelas/qcurious-tester)",
    )
    args = parser.parse_args()

    api = HfApi()

    # Create the dataset repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # Upload dataset card (README.md)
    card_path = Path(__file__).resolve().parent / "hf_dataset_card.md"
    if card_path.exists():
        print("  Uploading dataset card (README.md)...")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    uploaded, skipped = 0, 0
    for rel_path in RESULT_FILES:
        local_path = RESULTS_DIR / rel_path
        if not local_path.exists():
            print(f"  SKIP (not found): {rel_path}")
            skipped += 1
            continue

        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  Uploading {rel_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        uploaded += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
