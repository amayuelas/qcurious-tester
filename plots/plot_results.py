"""Generate paper figures from benchmark results.

Usage:
    python plots/plot_results.py results/repo_explore_bench/reb_quick_test.json
    python plots/plot_results.py results/testgeneval/tge_quick_test_v4.json
    python plots/plot_results.py results/repo_explore_bench/*.json results/testgeneval/*.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

STRATEGY_COLORS = {
    "random": "#888888",
    "greedy": "#E8A84C",
    "cov_greedy": "#4C9BE8",
    "cov_qvalue": "#E84C4C",
    "cov_planned": "#4CE88A",
}

STRATEGY_LABELS = {
    "random": "Random",
    "cov_greedy": "CovGreedy",
    "cov_qvalue": "CovQValue (ours)",
    "cov_planned": "CovPlanned",
    "greedy": "Greedy",
}


def load_results(paths):
    """Load and merge results from one or more JSON files."""
    all_results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for r in data["results"]:
            r["_source"] = str(p)
        all_results.extend(data["results"])
    return all_results


def get_strategies(results):
    """Get strategy names from results."""
    strats = set()
    for r in results:
        strats.update(r["strategies"].keys())
    return sorted(strats, key=lambda s: list(STRATEGY_COLORS.keys()).index(s)
                  if s in STRATEGY_COLORS else 99)


# ---------------------------------------------------------------------------
# Figure 1: Exploration curves (coverage vs step)
# ---------------------------------------------------------------------------

def plot_exploration_curves(results, output_path, metric="branch"):
    """Plot average coverage vs execution step for each strategy.

    This is the key figure — shows exploration dynamics.
    """
    strategies = get_strategies(results)
    curve_key = "branch_curve" if metric == "branch" else "line_curve"

    # Collect curves, pad to same length
    curves_by_strategy = defaultdict(list)
    for r in results:
        for s in strategies:
            if s in r["strategies"]:
                curve = r["strategies"][s].get(curve_key, [])
                if curve:
                    curves_by_strategy[s].append(curve)

    if not curves_by_strategy:
        print(f"No {curve_key} data found in results")
        return

    # Find max length
    max_len = max(len(c) for curves in curves_by_strategy.values()
                  for c in curves)

    fig, ax = plt.subplots(figsize=(8, 5))

    for s in strategies:
        curves = curves_by_strategy.get(s, [])
        if not curves:
            continue

        # Pad shorter curves with their last value
        padded = []
        for c in curves:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            padded.append(c[:max_len])

        arr = np.array(padded)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)

        steps = np.arange(1, max_len + 1)
        color = STRATEGY_COLORS.get(s, "#000000")
        label = STRATEGY_LABELS.get(s, s)

        ax.plot(steps, mean, color=color, linewidth=2, label=label)
        if len(arr) > 1:
            ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.15)

    ax.set_xlabel("Execution Step")
    ax.set_ylabel(f"Cumulative {'Branch' if metric == 'branch' else 'Line'} Coverage")
    ax.set_title(f"Exploration Dynamics ({len(results)} targets)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_len)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-target comparison (bar chart)
# ---------------------------------------------------------------------------

def plot_per_target_bars(results, output_path):
    """Bar chart comparing strategies per target."""
    strategies = get_strategies(results)

    # Get unique targets
    targets = []
    finals = defaultdict(dict)
    for r in results:
        label = r.get("module", r.get("code_file", "?"))
        # Shorten long module names
        parts = label.split(".")
        if len(parts) > 2:
            label = ".".join(parts[-2:])
        if label not in targets:
            targets.append(label)
        for s in strategies:
            if s in r["strategies"]:
                finals[label][s] = r["strategies"][s].get("final", 0)

    if len(targets) > 20:
        # Too many targets — show top 20 by max coverage
        targets.sort(key=lambda t: max(finals[t].values()) if finals[t] else 0,
                     reverse=True)
        targets = targets[:20]

    x = np.arange(len(targets))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(max(10, len(targets) * 0.8), 5))

    for i, s in enumerate(strategies):
        vals = [finals[t].get(s, 0) for t in targets]
        color = STRATEGY_COLORS.get(s, "#000000")
        label = STRATEGY_LABELS.get(s, s)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=color, alpha=0.85)

    ax.set_xlabel("Target Module")
    ax.set_ylabel("Branch Coverage")
    ax.set_title(f"Per-Target Branch Coverage ({len(targets)} targets)")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Pass rate comparison
# ---------------------------------------------------------------------------

def plot_pass_rates(results, output_path):
    """Bar chart of pass rates per strategy."""
    strategies = get_strategies(results)

    fig, ax = plt.subplots(figsize=(6, 4))

    means = []
    ses = []
    colors = []
    labels = []
    for s in strategies:
        rates = [r["strategies"][s].get("pass_rate", 0) for r in results
                 if s in r["strategies"]]
        if rates:
            means.append(np.mean(rates) * 100)
            ses.append(np.std(rates) / np.sqrt(len(rates)) * 100 if len(rates) > 1 else 0)
        else:
            means.append(0)
            ses.append(0)
        colors.append(STRATEGY_COLORS.get(s, "#000000"))
        labels.append(STRATEGY_LABELS.get(s, s))

    x = np.arange(len(strategies))
    ax.bar(x, means, yerr=ses, color=colors, alpha=0.85, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Test Script Pass Rate by Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Per-repo breakdown
# ---------------------------------------------------------------------------

def plot_per_repo(results, output_path):
    """Grouped bar chart of mean coverage per repo per strategy."""
    strategies = get_strategies(results)

    repos = sorted(set(r.get("repo", "unknown") for r in results))
    if len(repos) <= 1:
        return  # Not useful with 1 repo

    repo_means = {}
    for repo in repos:
        repo_results = [r for r in results if r.get("repo") == repo]
        repo_means[repo] = {}
        for s in strategies:
            vals = [r["strategies"][s].get("final", 0) for r in repo_results
                    if s in r["strategies"]]
            repo_means[repo][s] = np.mean(vals) if vals else 0

    x = np.arange(len(repos))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(max(8, len(repos) * 1.2), 5))

    for i, s in enumerate(strategies):
        vals = [repo_means[repo].get(s, 0) for repo in repos]
        color = STRATEGY_COLORS.get(s, "#000000")
        label = STRATEGY_LABELS.get(s, s)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=color, alpha=0.85)

    # Shorten repo names
    short_repos = [r.split("/")[-1] if "/" in r else r for r in repos]
    ax.set_xticks(x)
    ax.set_xticklabels(short_repos, rotation=45, ha="right")
    ax.set_ylabel("Mean Branch Coverage")
    ax.set_title("Coverage by Repository")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("results", nargs="+", help="Result JSON file(s)")
    parser.add_argument("--output-dir", default="plots", help="Output directory")
    parser.add_argument("--prefix", default="", help="Filename prefix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    prefix = args.prefix + "_" if args.prefix else ""

    results = load_results(args.results)
    print(f"Loaded {len(results)} runs from {len(args.results)} file(s)")

    strategies = get_strategies(results)
    print(f"Strategies: {strategies}")

    # Generate all plots
    plot_exploration_curves(results, output_dir / f"{prefix}exploration_curves.png",
                           metric="branch")
    plot_exploration_curves(results, output_dir / f"{prefix}exploration_curves_lines.png",
                           metric="line")
    plot_per_target_bars(results, output_dir / f"{prefix}per_target.png")
    plot_pass_rates(results, output_dir / f"{prefix}pass_rates.png")
    plot_per_repo(results, output_dir / f"{prefix}per_repo.png")

    print("Done!")


if __name__ == "__main__":
    main()
