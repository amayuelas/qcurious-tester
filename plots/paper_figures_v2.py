"""Figures and main-table numbers for the resubmission (final_* runs).

Reuses the styling and plotting code of paper_figures.py, but loads the
resubmission result files: RepoExploreBench from final_reb_<model>[_s<seed>].json
and TestGenEval from the per-repo files in testgeneval/by_repo/<model>/ (these
carry the zero-coverage repairs; full_run_<model>.json does not). The method is
stored as "covqvalue2" and is renamed to "cov_qvalue" here. Runs with several
seeds are averaged per target before pooling, so every model weighs the same.

Usage (from the repo root):
    python plots/paper_figures_v2.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import paper_figures as pf  # noqa: E402

RESULTS = Path("results")
MODELS = {"gemini38": "Gemini 3.8 Flash", "gemma4": "Gemma 4 31B", "glm53": "GLM 5.3"}
REB_PREFIX = {"gemini38": "gemini38", "gemma4": "gemma", "glm53": "glm53"}
RENAME = {"covqvalue2": "cov_qvalue"}
TGE_TARGETS = 140
STRATEGIES = pf.STRATEGIES


def _normalize(results):
    for r in results:
        r["strategies"] = {RENAME.get(s, s): v for s, v in r["strategies"].items()}
    return results


def _average_seeds(runs):
    """Average final and branch_curve per target across seed runs."""
    by_target = {}
    for results in runs:
        for r in results:
            by_target.setdefault(r["module"], []).append(r)
    out = []
    for module, rs in by_target.items():
        merged = {"module": module, "repo": rs[0].get("repo"), "strategies": {}}
        for s in rs[0]["strategies"]:
            vals = [r["strategies"][s] for r in rs if s in r["strategies"]]
            merged["strategies"][s] = {
                "final": float(np.mean([v["final"] for v in vals])),
                "branch_curve": list(np.mean([v["branch_curve"] for v in vals], axis=0)),
                "pass_rate": float(np.mean([v.get("pass_rate", 0) for v in vals])),
                "final_lines": float(np.mean([v.get("final_lines", 0) for v in vals])),
            }
        out.append(merged)
    return out


def load():
    data = {"repo_explore_bench": {}, "testgeneval": {}}
    reb = RESULTS / "repo_explore_bench"
    for mk in MODELS:
        pre = REB_PREFIX[mk]
        files = sorted(glob.glob(str(reb / f"final_reb_{pre}.json"))
                       + glob.glob(str(reb / f"final_reb_{pre}_s*.json")))
        if files:
            runs = [_normalize(json.load(open(f))["results"]) for f in files]
            results = runs[0] if len(runs) == 1 else _average_seeds(runs)
            data["repo_explore_bench"][mk] = {"results": results}
        tge_files = sorted(glob.glob(str(RESULTS / "testgeneval" / "by_repo" / mk / "*.json")))
        if tge_files:
            results = []
            for f in tge_files:
                results += json.load(open(f))["results"]
            # only complete runs: a model still running would bias every table
            if len(results) >= TGE_TARGETS:
                data["testgeneval"][mk] = {"results": _normalize(results)}
            else:
                print(f"skipping {mk} TestGenEval: {len(results)}/{TGE_TARGETS} targets")
    return data

# Layout-adjusted copies of paper_figures functions (legend/label placement).
# They use pf's styling globals through these aliases.
plt = pf.plt
STRATEGY_LABELS, STRATEGY_COLORS, PLOTS_DIR = pf.STRATEGY_LABELS, pf.STRATEGY_COLORS, pf.PLOTS_DIR


def plot_per_repo(data, bench, model_keys, title_suffix, filename):
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
               label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s],
               alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([short[r] for r in repos], rotation=45, ha="right")
    ax.set_ylabel("Mean Branch Coverage")
    # legend above the axes so it cannot cover a bar
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    ax.grid(True, alpha=0.4, axis="y", linewidth=1.0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{filename}.pdf")
    plt.savefig(PLOTS_DIR / f"{filename}.png")
    plt.close()
    print(f"Saved {filename}")


def plot_pass_rate_vs_coverage(data):
    """Pass rate vs coverage: transparent dots per (model,bench), bold means."""
    # Larger sizes for 50% textwidth display
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.tick_params(labelsize=16)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

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
                          s=250, alpha=0.3, edgecolors="none", zorder=2)

    # Grand mean per strategy: large opaque dot with label
    label_offsets = {
        "random": (20, -8),
        "greedy": (-85, -5),
        "cov_greedy": (-50, -48),
        "cov_qvalue": (14, 8),
    }
    for s in STRATEGIES:
        if not grand[s]["pr"]:
            continue
        mean_pr = np.mean(grand[s]["pr"])
        mean_br = np.mean(grand[s]["br"])
        ax.scatter(mean_pr, mean_br, color=STRATEGY_COLORS[s],
                  s=600, alpha=1.0, edgecolors="black", linewidths=2.0,
                  zorder=4, label=STRATEGY_LABELS[s])
        ax.annotate(STRATEGY_LABELS[s], (mean_pr, mean_br),
                   textcoords="offset points",
                   xytext=label_offsets.get(s, (12, 7)),
                   fontsize=17, fontweight="bold" if s == "cov_qvalue" else "normal",
                   zorder=5)

    ax.set_xlabel("Pass Rate (%)")
    ax.set_ylabel("Mean Branch Coverage")
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_pass_rate_vs_coverage.pdf")
    plt.savefig(PLOTS_DIR / "fig4_pass_rate_vs_coverage.png")
    plt.close()
    print("Saved fig4_pass_rate_vs_coverage")


def main_table_stats(data):
    """Mean ± SE per strategy, and CovQValue vs Greedy (Δ, Cohen's d, win rate, p)."""
    for bench in data:
        for mk, d in data[bench].items():
            R = d["results"]
            print(f"\n{bench} / {mk}  (n={len(R)})")
            for s in STRATEGIES:
                v = np.array([r["strategies"][s]["final"] for r in R])
                print(f"  {s:12s} {v.mean():6.1f} ± {v.std(ddof=1) / np.sqrt(len(v)):4.1f}")
            cq = np.array([r["strategies"]["cov_qvalue"]["final"] for r in R])
            gr = np.array([r["strategies"]["greedy"]["final"] for r in R])
            diff = cq - gr
            print(f"  Δ vs greedy {diff.mean():+.1f}  d={diff.mean() / diff.std(ddof=1):.2f}  "
                  f"win={np.mean(diff > 0):.0%}  p={stats.ttest_rel(cq, gr).pvalue:.1e}")


def _bold_max(vals, fmt):
    top = max(vals)
    return [f"\\textbf{{{fmt(v)}}}" if v == top else fmt(v) for v in vals]


def latex_appendix_tables(data):
    """Row bodies for the appendix tables (line coverage/pass rate, per repo).
    Models without results for a benchmark get "--" cells."""
    labels = {"random": "Random", "greedy": "Greedy",
              "cov_greedy": "CovGreedy", "cov_qvalue": "CovQValue"}
    print("\n% --- tab:lines ---")
    for bench, name in [("repo_explore_bench", "RepoExploreBench"),
                        ("testgeneval", "TestGenEval Lite")]:
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{name}}}}} \\\\")
        lines = {}
        for mk in MODELS:
            if mk in data[bench]:
                R = data[bench][mk]["results"]
                lines[mk] = {s: np.mean([r["strategies"][s]["final_lines"] for r in R])
                             for s in STRATEGIES}
        for s in STRATEGIES:
            cells = []
            for mk in MODELS:
                if mk not in data[bench]:
                    cells += ["--", "--"]
                    continue
                R = data[bench][mk]["results"]
                ln = lines[mk][s]
                pr = 100 * np.mean([r["strategies"][s]["pass_rate"] for r in R])
                best = ln == max(lines[mk].values())
                cells += [f"\\textbf{{{ln:.0f}}}" if best else f"{ln:.0f}", f"{pr:.0f}\\%"]
            print(f"\\quad {labels[s]} & " + " & ".join(cells) + " \\\\")
        if bench == "repo_explore_bench":
            print("\\midrule")

    for bench, order in [("testgeneval", ["astropy", "django", "matplotlib", "seaborn",
                                          "flask", "xarray", "pylint", "pytest",
                                          "sphinx", "sympy"]),
                         ("repo_explore_bench", None)]:
        print(f"\n% --- per repo: {bench} ---")
        repos = order or sorted({r["repo"] for mk in data[bench]
                                 for r in data[bench][mk]["results"]})
        rows = {rp: [] for rp in repos}
        mean_row = []
        for mk in MODELS:
            if mk not in data[bench]:
                for rp in repos:
                    rows[rp] += ["--"] * 4
                mean_row += ["--"] * 4
                continue
            R = data[bench][mk]["results"]
            for rp in repos:
                sub = [r for r in R if r["repo"].split("/")[-1] == rp]
                rows[rp] += _bold_max([round(np.mean([r["strategies"][s]["final"] for r in sub]))
                                       for s in STRATEGIES], lambda v: f"{v:.0f}")
            mean_row += _bold_max([np.mean([r["strategies"][s]["final"] for r in R])
                                   for s in STRATEGIES], lambda v: f"{v:.1f}")
        for rp in repos:
            print(f"{rp} & " + " & ".join(rows[rp]) + " \\\\")
        print("\\midrule")
        print("Mean & " + " & ".join(mean_row) + " \\\\")


def _key(r):
    # TestGenEval holds the same file at several repo versions
    return (r["module"], str(r.get("version", "")))


def best_observed(bench):
    """Per target: highest final count reached by any strategy, model or seed
    in the main runs (the rebuttal's "max-achieved" denominator)."""
    files = (glob.glob(str(RESULTS / "repo_explore_bench" / "final_reb_*.json"))
             if bench == "repo_explore_bench"
             else [f for mk in MODELS
                   for f in glob.glob(str(RESULTS / "testgeneval" / "by_repo" / mk / "*.json"))
                   if mk in COMPLETE_TGE])
    best = {}
    for f in files:
        for r in json.load(open(f))["results"]:
            for v in r["strategies"].values():
                best[_key(r)] = max(best.get(_key(r), 0), v["final"])
    return best


COMPLETE_TGE = set()


def latex_pct_table(data):
    """Rows for the appendix table: mean per-target coverage as a percentage of
    the best coverage observed on that target."""
    print("\n% --- tab:pct (percent of best observed) ---")
    labels = {"random": "Random", "greedy": "Greedy",
              "cov_greedy": "CovGreedy", "cov_qvalue": "CovQValue"}
    COMPLETE_TGE.update(data["testgeneval"])
    cols = {}
    for bench in ("repo_explore_bench", "testgeneval"):
        best = best_observed(bench)
        for mk in MODELS:
            if mk not in data[bench]:
                cols[(bench, mk)] = None
                continue
            R = [r for r in data[bench][mk]["results"] if best.get(_key(r), 0) > 0]
            cols[(bench, mk)] = {
                s: (np.mean([100 * r["strategies"][s]["final"] / best[_key(r)] for r in R]),
                    np.std([100 * r["strategies"][s]["final"] / best[_key(r)] for r in R],
                           ddof=1) / np.sqrt(len(R)))
                for s in STRATEGIES}
    for s in STRATEGIES:
        cells = []
        for c in cols.values():
            if c is None:
                cells.append("--")
                continue
            m, se = c[s]
            top = max(v[0] for v in c.values()) == m
            val = f"\\textbf{{{m:.1f}\\%}}" if top else f"{m:.1f}\\%"
            cells.append(f"{val} {{\\tiny$\\pm${se:.1f}}}")
        print(f"{labels[s]} & " + " & ".join(cells) + " \\\\")


def main():
    data = load()
    # pf's figure functions read the model list from this module global
    pf.MODELS = {mk: (name, None) for mk, name in MODELS.items()}
    for bench, short in [("repo_explore_bench", "reb"), ("testgeneval", "tge")]:
        keys = list(data[bench])
        print(f"{bench}: models {keys}")
        # Figure 2: exploration curves, all models averaged
        pf._plot_exploration_curves_single(
            data, bench, keys, f"fig1_exploration_curves_{short}",
            title={"reb": "RepoExploreBench", "tge": "TestGenEval Lite"}[short])
        # Appendix: exploration curves per model
        for mk in keys:
            pf._plot_exploration_curves_single(
                data, bench, [mk], f"fig1_exploration_curves_{short}_{mk}",
                figsize=(4.5, 4), title=None, show_legend=(mk == keys[0]))
        # Figure 3 / appendix: per-repo coverage, all models averaged
        plot_per_repo(data, bench, keys, "all models", f"fig3_per_repo_{short}")
    # Figure 4: pass rate vs coverage
    plot_pass_rate_vs_coverage(data)
    main_table_stats(data)
    latex_appendix_tables(data)
    latex_pct_table(data)


if __name__ == "__main__":
    main()
