"""Ablation figures for the resubmission (Gemma 4 31B, RepoExploreBench, seed 42).

Reuses the styling of paper_ablations.py. Budgets 8/16/32 come from
ablation_budget<N>_gemma.json. N=24 is the default setting, so it is not rerun:
CovQValue comes from the full-method row of ablation_components_gemma.json (same
code and settings as the other ablations) and Random from the main Gemma run.

Usage (from the repo root):
    python plots/paper_ablations_v2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paper_ablations as pa  # noqa: E402  (sets the matplotlib style)

plt = pa.plt
ABL = Path("results/ablations")
METHOD = "cov_qvalue_tgt_fnt_bayes"


def finals(path, strategy):
    return [r["strategies"][strategy]["final"] for r in json.load(open(path))["results"]]


def fig_budget():
    qv, rn = {}, {}
    for b in (8, 16, 32):
        qv[b] = finals(ABL / f"ablation_budget{b}_gemma.json", METHOD)
        rn[b] = finals(ABL / f"ablation_budget{b}_gemma.json", "random")
    qv[24] = finals(ABL / "ablation_components_gemma.json", METHOD)
    rn[24] = finals(Path("results/repo_explore_bench/final_reb_gemma_s42.json"), "random")
    budgets = sorted(qv)
    qv_m = [np.mean(qv[b]) for b in budgets]
    qv_se = [np.std(qv[b]) / np.sqrt(len(qv[b])) for b in budgets]
    rn_m = [np.mean(rn[b]) for b in budgets]
    print("budget", budgets, "CovQValue", [round(x, 1) for x in qv_m],
          "Random", [round(x, 1) for x in rn_m])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(labelsize=22)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    ax.plot(budgets, qv_m, "o-", color=pa.COV_QVALUE_COLOR, linewidth=4.5,
            markersize=14, label="CovQValue", zorder=3)
    ax.fill_between(budgets, [m - s for m, s in zip(qv_m, qv_se)],
                    [m + s for m, s in zip(qv_m, qv_se)],
                    color=pa.COV_QVALUE_COLOR, alpha=0.15)
    ax.plot(budgets, rn_m, "s--", color=pa.RANDOM_COLOR, linewidth=4,
            markersize=12, label="Random", zorder=3)
    # CoverUp with the same model (Gemma 4 31B) runs to completion rather than
    # N executions, so it is a flat reference line. Pynguin is left out: it
    # sits on the Random line and is reported in the external-tools table.
    coverup = np.mean(finals(Path("results/repo_explore_bench/final_coverup_gemma.json"),
                             "coverup"))
    print(f"CoverUp (gemma) {coverup:.1f}")
    ax.axhline(coverup, color="#5B4E8C", linestyle=":", linewidth=4.5,
               label="CoverUp", zorder=2)

    ax.set_xlabel("Execution Budget $N$")
    ax.set_ylabel("Mean Branch Coverage")
    ax.legend(loc="upper left", fontsize=19, handlelength=1.6)
    ax.grid(True, alpha=0.5, linewidth=1.5)
    ax.set_xticks(budgets)
    ax.set_ylim(20, 138)
    plt.tight_layout()
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_budget.pdf")
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_budget.png")
    plt.close()
    print("Saved fig_ablation_budget")


def fig_plan_length():
    """Plan length S in {1, 3, 5}, two ways: at a fixed budget of N=24
    executions (fewer feedback rounds for longer plans), and at a fixed 8
    rounds (N = 8S, longer plans also get more executions). S=3 is the
    default setting (components run) in both. The fixed-rounds bars are drawn
    once ablation_Smatched{1,5}_gemma.json exist."""
    S = [1, 3, 5]
    default = finals(ABL / "ablation_components_gemma.json", METHOD)
    fixed_budget = {1: finals(ABL / "ablation_S1_gemma.json", METHOD), 3: default,
                    5: finals(ABL / "ablation_S5_gemma.json", METHOD)}
    series = [("Fixed budget ($N$=24)", fixed_budget, pa.COV_QVALUE_COLOR, None)]
    matched = {s: ABL / f"ablation_Smatched{s}_gemma.json" for s in (1, 5)}
    if all(f.exists() for f in matched.values()):
        rounds = {1: finals(matched[1], METHOD), 3: default, 5: finals(matched[5], METHOD)}
        series.append(("Fixed rounds ($N$=8$S$)", rounds, "#E8A9AB", "//"))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(labelsize=22)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    x = np.arange(len(S))
    width = 0.7 / len(series)
    top = 0
    for i, (label, data, color, hatch) in enumerate(series):
        m = [np.mean(data[s]) for s in S]
        se = [np.std(data[s]) / np.sqrt(len(data[s])) for s in S]
        top = max(top, max(a + b for a, b in zip(m, se)))
        print(label, [round(v, 1) for v in m])
        ax.bar(x + (i - (len(series) - 1) / 2) * width, m, width, yerr=se,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
               capsize=6, label=label, error_kw={"linewidth": 2, "capthick": 2})
    ax.set_ylim(0, top + (22 if len(series) > 1 else 12))
    ax.set_xlabel("Plan Length $S$")
    ax.set_ylabel("Mean Branch Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in S], fontsize=22)
    ax.grid(True, alpha=0.5, axis="y", linewidth=1.5)
    ax.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_s_matched.pdf")
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_s_matched.png")
    plt.close()
    print("Saved fig_ablation_s_matched")


def fig_num_plans():
    """Number of candidate plans K in {1, 3, 5} (N=24, S=3). K=3 is the
    default setting (components run)."""
    K = [1, 3, 5]
    data = {1: finals(ABL / "ablation_K1_gemma.json", METHOD),
            3: finals(ABL / "ablation_components_gemma.json", METHOD),
            5: finals(ABL / "ablation_K5_gemma.json", METHOD)}
    m = [np.mean(data[k]) for k in K]
    se = [np.std(data[k]) / np.sqrt(len(data[k])) for k in K]
    print("K", K, [round(v, 1) for v in m])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(labelsize=22)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    x = np.arange(len(K))
    ax.bar(x, m, 0.5, yerr=se, color=pa.COV_QVALUE_COLOR, alpha=0.88,
           capsize=8, edgecolor="white", linewidth=0.5,
           error_kw={"linewidth": 2.5, "capthick": 2.5})
    ax.set_ylim(0, max(a + b for a, b in zip(m, se)) + 12)
    ax.set_xlabel("Number of Plans $K$")
    ax.set_ylabel("Mean Branch Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K], fontsize=22)
    ax.grid(True, alpha=0.5, axis="y", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_K.pdf")
    plt.savefig(pa.PLOTS_DIR / "fig_ablation_K.png")
    plt.close()
    print("Saved fig_ablation_K")


def table_rows():
    """Values for the ablation table: gamma sweep and component decomposition."""
    g = {k: np.mean(finals(ABL / "ablation_gamma_gemma.json", f"{METHOD}_{k}"))
         for k in ("g0", "g025", "g05", "g1")}
    print("gamma 0 / 0.25 / 0.5 / 1:", " / ".join(f"{g[k]:.1f}" for k in g))
    for s in ("cov_greedy", "divhints_random", "cov_qvalue_tgt_rnd",
              "cov_qvalue_tgt_fnt", METHOD):
        v = finals(ABL / "ablation_components_gemma.json", s)
        print(f"  {s:28s} {np.mean(v):.1f} ± {np.std(v) / np.sqrt(len(v)):.1f}")


if __name__ == "__main__":
    fig_budget()
    fig_plan_length()
    fig_num_plans()
    table_rows()
