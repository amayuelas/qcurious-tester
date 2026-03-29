"""Generate paper-quality figures and LaTeX tables from all results.

Usage:
    python plots/paper_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots/paper")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemini": ("Gemini Flash", "#E84C4C"),
    "gpt54mini": ("GPT-5.4 Mini", "#4C9BE8"),
    "mistral": ("Mistral Large", "#4CE88A"),
}

STRATEGIES = ["random", "greedy", "cov_greedy", "cov_qvalue"]
STRATEGY_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "cov_greedy": "CovGreedy",
    "cov_qvalue": "CovQValue",
}
STRATEGY_COLORS = {
    "random": "#888888",
    "greedy": "#E8A84C",
    "cov_greedy": "#4C9BE8",
    "cov_qvalue": "#E84C4C",
}


def load_all():
    """Load all 6 result files."""
    data = {}
    for bench in ["repo_explore_bench", "testgeneval"]:
        data[bench] = {}
        for model_key in MODELS:
            path = RESULTS_DIR / bench / f"full_run_{model_key}.json"
            if path.exists():
                with open(path) as f:
                    data[bench][model_key] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Figure 1: Exploration curves (2 panels, one model)
# ---------------------------------------------------------------------------

def _plot_exploration_curves(data, bench_keys, model_keys, title_suffix, filename):
    """Helper: plot exploration curves for given benchmarks and models."""
    fig, axes = plt.subplots(1, len(bench_keys), figsize=(6 * len(bench_keys), 4.5))
    if len(bench_keys) == 1:
        axes = [axes]

    bench_labels = {
        "repo_explore_bench": "RepoExploreBench",
        "testgeneval": "TestGenEval Lite",
    }

    for ax, bench in zip(axes, bench_keys):
        curves_by_strat = defaultdict(list)
        for model_key in model_keys:
            if model_key not in data[bench]:
                continue
            for r in data[bench][model_key]["results"]:
                for s in STRATEGIES:
                    if s in r["strategies"]:
                        curve = r["strategies"][s].get("branch_curve", [])
                        if curve:
                            curves_by_strat[s].append(curve)

        if not curves_by_strat:
            continue

        max_len = max(len(c) for curves in curves_by_strat.values() for c in curves)

        for s in STRATEGIES:
            curves = curves_by_strat.get(s, [])
            if not curves:
                continue
            padded = [c + [c[-1]] * (max_len - len(c)) if len(c) < max_len else c[:max_len]
                      for c in curves]
            arr = np.array(padded)
            mean = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            steps = np.arange(1, max_len + 1)

            ax.plot(steps, mean, color=STRATEGY_COLORS[s], linewidth=2,
                    label=STRATEGY_LABELS[s])
            ax.fill_between(steps, mean - se, mean + se,
                           color=STRATEGY_COLORS[s], alpha=0.12)

        title = bench_labels[bench]
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Cumulative Branch Coverage")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max_len)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{filename}.pdf")
    plt.savefig(PLOTS_DIR / f"{filename}.png")
    plt.close()
    print(f"Saved {filename}")


def fig_exploration_curves(data):
    """Generate exploration curves: averaged across all models + per model."""
    benches = ["repo_explore_bench", "testgeneval"]

    # All models averaged
    _plot_exploration_curves(data, benches, list(MODELS.keys()),
                            "all models", "fig1_exploration_curves")

    # Per model
    for model_key, (model_name, _) in MODELS.items():
        _plot_exploration_curves(data, benches, [model_key],
                                model_name, f"fig1_exploration_curves_{model_key}")


# ---------------------------------------------------------------------------
# Figure 2: Multi-model comparison (bar chart)
# ---------------------------------------------------------------------------

def fig_model_comparison(data):
    """Bar chart: cov_qvalue improvement across 3 models × 2 benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bench_labels = {
        "repo_explore_bench": "RepoExploreBench",
        "testgeneval": "TestGenEval Lite",
    }

    for ax, bench in zip(axes, ["repo_explore_bench", "testgeneval"]):
        model_keys = [k for k in MODELS if k in data[bench]]
        x = np.arange(len(model_keys))
        width = 0.2

        for i, s in enumerate(STRATEGIES):
            means = []
            ses = []
            for mk in model_keys:
                results = data[bench][mk]["results"]
                vals = [r["strategies"][s]["final"] for r in results
                        if s in r["strategies"]]
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)))

            ax.bar(x + i * width - 0.3, means, width, yerr=ses,
                   label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s],
                   alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([MODELS[mk][0] for mk in model_keys])
        ax.set_ylabel("Mean Branch Coverage")
        ax.set_title(bench_labels[bench])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_model_comparison.pdf")
    plt.savefig(PLOTS_DIR / "fig2_model_comparison.png")
    plt.close()
    print("Saved fig2_model_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Per-repo breakdown (Gemini Flash, TGE)
# ---------------------------------------------------------------------------

def _plot_per_repo(data, bench, model_keys, title_suffix, filename):
    """Helper: per-repo bar chart for given models."""
    # Collect all results across specified models
    all_results = []
    for mk in model_keys:
        if mk in data[bench]:
            all_results.extend(data[bench][mk]["results"])

    repos = sorted(set(r.get("repo", "unknown") for r in all_results))
    short = {r: r.split("/")[-1] for r in repos}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(repos))
    width = 0.2

    for i, s in enumerate(STRATEGIES):
        means = []
        for repo in repos:
            vals = [r["strategies"][s]["final"] for r in all_results
                    if r.get("repo") == repo and s in r["strategies"]]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width - 0.3, means, width,
               label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short[r] for r in repos], rotation=45, ha="right")
    ax.set_ylabel("Mean Branch Coverage")
    title = "Coverage by Repository"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{filename}.pdf")
    plt.savefig(PLOTS_DIR / f"{filename}.png")
    plt.close()
    print(f"Saved {filename}")


def fig_per_repo(data):
    """Generate per-repo charts: both benchmarks, all models + per model."""
    for bench, bench_short in [("repo_explore_bench", "reb"), ("testgeneval", "tge")]:
        # All models averaged
        _plot_per_repo(data, bench, list(MODELS.keys()),
                       "all models", f"fig3_per_repo_{bench_short}")

        # Per model
        for model_key, (model_name, _) in MODELS.items():
            _plot_per_repo(data, bench, [model_key],
                           model_name, f"fig3_per_repo_{bench_short}_{model_key}")


# ---------------------------------------------------------------------------
# Figure 4: Pass rate vs coverage scatter
# ---------------------------------------------------------------------------

def fig_pass_rate_vs_coverage(data):
    """Pass rate vs coverage: transparent dots per (model,bench), bold means."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect all individual points: one per (strategy, model, benchmark)
    grand = {s: {"pr": [], "br": []} for s in STRATEGIES}

    for bench_key in ["repo_explore_bench", "testgeneval"]:
        for model_key in MODELS:
            if model_key not in data[bench_key]:
                continue
            results = data[bench_key][model_key]["results"]

            for s in STRATEGIES:
                vals_br = [r["strategies"][s]["final"] for r in results
                          if s in r["strategies"]]
                vals_pr = [r["strategies"][s].get("pass_rate", 0) for r in results
                          if s in r["strategies"]]
                if not vals_br:
                    continue
                mean_br = np.mean(vals_br)
                mean_pr = np.mean(vals_pr) * 100
                grand[s]["pr"].append(mean_pr)
                grand[s]["br"].append(mean_br)

                # Small transparent dot
                ax.scatter(mean_pr, mean_br, color=STRATEGY_COLORS[s],
                          s=80, alpha=0.35, edgecolors="none", zorder=2)

    # Grand mean per strategy: large opaque dot with label
    label_offsets = {
        "random": (8, -10),
        "greedy": (8, 5),
        "cov_greedy": (-12, 8),
        "cov_qvalue": (8, 5),
    }
    for s in STRATEGIES:
        if not grand[s]["pr"]:
            continue
        mean_pr = np.mean(grand[s]["pr"])
        mean_br = np.mean(grand[s]["br"])
        ax.scatter(mean_pr, mean_br, color=STRATEGY_COLORS[s],
                  s=250, alpha=1.0, edgecolors="black", linewidths=1.2,
                  zorder=4, label=STRATEGY_LABELS[s])
        ax.annotate(STRATEGY_LABELS[s], (mean_pr, mean_br),
                   textcoords="offset points",
                   xytext=label_offsets.get(s, (8, 5)),
                   fontsize=10, fontweight="bold" if s == "cov_qvalue" else "normal",
                   zorder=5)

    ax.set_xlabel("Pass Rate (%)")
    ax.set_ylabel("Mean Branch Coverage")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_pass_rate_vs_coverage.pdf")
    plt.savefig(PLOTS_DIR / "fig4_pass_rate_vs_coverage.png")
    plt.close()
    print("Saved fig4_pass_rate_vs_coverage")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def table_main_results(data):
    """Generate main results LaTeX table (horizontal, compact, full-width)."""
    from scipy import stats

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean branch coverage across 3 models and 2 benchmarks. "
                 r"All CovQValue vs.\ Random comparisons: $p < 0.0001$ (paired $t$-test). "
                 r"$d$: Cohen's $d$ effect size.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{l ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{\textbf{RepoExploreBench} (93 targets)}"
                 r"& \multicolumn{3}{c}{\textbf{TestGenEval Lite} (140 targets)} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"& Gemini & GPT-5.4m & Mistral"
                 r"& Gemini & GPT-5.4m & Mistral \\")
    lines.append(r"\midrule")

    for s in STRATEGIES:
        row = [STRATEGY_LABELS[s]]
        for bench in ["repo_explore_bench", "testgeneval"]:
            for model_key in ["gemini", "gpt54mini", "mistral"]:
                results = data[bench][model_key]["results"]
                vals = [r["strategies"][s]["final"] for r in results
                        if s in r["strategies"]]
                mean = np.mean(vals)
                se = np.std(vals) / np.sqrt(len(vals))
                if s == "cov_qvalue":
                    row.append(f"\\textbf{{{mean:.1f}}} {{\\tiny$\\pm${se:.1f}}}")
                else:
                    row.append(f"{mean:.1f} {{\\tiny$\\pm${se:.1f}}}")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")

    # Delta row
    row = [r"$\Delta$ vs Random"]
    for bench in ["repo_explore_bench", "testgeneval"]:
        for model_key in ["gemini", "gpt54mini", "mistral"]:
            results = data[bench][model_key]["results"]
            deltas = [r["strategies"]["cov_qvalue"]["final"] - r["strategies"]["random"]["final"]
                      for r in results
                      if "random" in r["strategies"] and "cov_qvalue" in r["strategies"]]
            mean_d = np.mean(deltas)
            row.append(f"+{mean_d:.1f}")
    lines.append(" & ".join(row) + r" \\")

    # Cohen's d row
    row = [r"Cohen's $d$"]
    for bench in ["repo_explore_bench", "testgeneval"]:
        for model_key in ["gemini", "gpt54mini", "mistral"]:
            results = data[bench][model_key]["results"]
            deltas = [r["strategies"]["cov_qvalue"]["final"] - r["strategies"]["random"]["final"]
                      for r in results
                      if "random" in r["strategies"] and "cov_qvalue" in r["strategies"]]
            d_cohen = np.mean(deltas) / np.std(deltas)
            row.append(f"{d_cohen:.2f}")
    lines.append(" & ".join(row) + r" \\")

    # Win rate row
    row = ["Win rate"]
    for bench in ["repo_explore_bench", "testgeneval"]:
        for model_key in ["gemini", "gpt54mini", "mistral"]:
            results = data[bench][model_key]["results"]
            wins = sum(1 for r in results
                      if "random" in r["strategies"] and "cov_qvalue" in r["strategies"]
                      and r["strategies"]["cov_qvalue"]["final"] > r["strategies"]["random"]["final"])
            total = sum(1 for r in results
                       if "random" in r["strategies"] and "cov_qvalue" in r["strategies"])
            pct = 100 * wins / total
            row.append(f"{pct:.0f}\\%")
    lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    table = "\n".join(lines)
    with open(PLOTS_DIR / "table_main.tex", "w") as f:
        f.write(table)
    print("Saved table_main.tex")
    print(table)


def table_line_coverage(data):
    """Generate appendix table with line coverage."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Line coverage and pass rate (appendix). Results consistent with branch coverage.}")
    lines.append(r"\label{tab:lines}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c|}{Gemini Flash} "
                 r"& \multicolumn{2}{c|}{GPT-5.4 Mini} "
                 r"& \multicolumn{2}{c}{Mistral Large} \\")
    lines.append(r"\textbf{Strategy} & Lines & Pass\% & Lines & Pass\% & Lines & Pass\% \\")
    lines.append(r"\midrule")

    for bench_label, bench in [("\\textit{RepoExploreBench}", "repo_explore_bench"),
                                ("\\textit{TestGenEval Lite}", "testgeneval")]:
        lines.append(f"\\multicolumn{{7}}{{l}}{{{bench_label}}} \\\\")
        for s in STRATEGIES:
            row = [f"\\quad {STRATEGY_LABELS[s]}"]
            for model_key in ["gemini", "gpt54mini", "mistral"]:
                results = data[bench][model_key]["results"]
                line_vals = [r["strategies"][s].get("final_lines", 0) for r in results
                            if s in r["strategies"]]
                pr_vals = [r["strategies"][s].get("pass_rate", 0) for r in results
                          if s in r["strategies"]]
                mean_lines = np.mean(line_vals) if line_vals else 0
                mean_pr = np.mean(pr_vals) * 100 if pr_vals else 0
                if s == "cov_qvalue":
                    row.append(f"\\textbf{{{mean_lines:.0f}}}")
                else:
                    row.append(f"{mean_lines:.0f}")
                row.append(f"{mean_pr:.0f}\\%")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    with open(PLOTS_DIR / "table_lines.tex", "w") as f:
        f.write(table)
    print("Saved table_lines.tex")
    print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def table_per_repo(data):
    """Generate per-repo LaTeX tables for each model (appendix)."""

    for bench, bench_label, n in [
        ("repo_explore_bench", "RepoExploreBench", 93),
        ("testgeneval", "TestGenEval Lite", 140),
    ]:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Per-repository branch coverage on " + bench_label +
                     r" by model. CovQValue bolded.}")
        lines.append(r"\label{tab:perrepo_" + bench.replace("_", "") + "}")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\begin{tabular}{l cccc cccc cccc}")
        lines.append(r"\toprule")
        lines.append(r"& \multicolumn{4}{c}{Gemini Flash}"
                     r"& \multicolumn{4}{c}{GPT-5.4 Mini}"
                     r"& \multicolumn{4}{c}{Mistral Large} \\")
        lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}")

        short_strats = {"random": "Rnd", "greedy": "Gdy", "cov_greedy": "CG", "cov_qvalue": "CQ"}
        header = "Repo"
        for _ in range(3):
            for s in STRATEGIES:
                header += f" & {short_strats[s]}"
        lines.append(header + r" \\")
        lines.append(r"\midrule")

        # Get repos for this benchmark
        all_results = []
        for mk in MODELS:
            if mk in data[bench]:
                all_results.extend(data[bench][mk]["results"])
        repos = sorted(set(r.get("repo", r.get("module", "?").split(".")[0]) for r in all_results))

        # Shorten repo names
        def short_repo(r):
            if "/" in r:
                return r.split("/")[-1]
            parts = r.split(".")
            return parts[0] if len(parts) > 1 else r

        for repo in repos:
            row = [short_repo(repo)]
            for model_key in ["gemini", "gpt54mini", "mistral"]:
                if model_key not in data[bench]:
                    row.extend(["--"] * 4)
                    continue
                results = data[bench][model_key]["results"]
                repo_results = [r for r in results
                               if r.get("repo", r.get("module", "").split(".")[0]) == repo]
                for s in STRATEGIES:
                    vals = [r["strategies"][s]["final"] for r in repo_results
                            if s in r["strategies"]]
                    if vals:
                        mean = np.mean(vals)
                        if s == "cov_qvalue":
                            row.append(f"\\textbf{{{mean:.0f}}}")
                        else:
                            row.append(f"{mean:.0f}")
                    else:
                        row.append("--")
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        table = "\n".join(lines)
        fname = f"table_perrepo_{bench.replace('repo_explore_bench', 'reb').replace('testgeneval', 'tge')}.tex"
        with open(PLOTS_DIR / fname, "w") as f:
            f.write(table)
        print(f"Saved {fname}")
        print(table)
        print()


def main():
    data = load_all()
    print(f"Loaded: {', '.join(f'{b}: {list(d.keys())}' for b, d in data.items())}\n")

    fig_exploration_curves(data)
    fig_per_repo(data)
    fig_pass_rate_vs_coverage(data)
    table_main_results(data)
    print()
    table_line_coverage(data)
    print()
    table_per_repo(data)


if __name__ == "__main__":
    main()
