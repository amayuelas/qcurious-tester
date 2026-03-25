"""Coverage curves and comparison tables (text-based)."""


def print_comparison_table(results: dict):
    """Print a summary comparison table."""
    print(f"\n{'Program':<22} {'Corr':<6} {'Random':<8} {'Greedy':<8} "
          f"{'Curiosity':<10} {'Blind':<8} {'Δ(C-G)':<8} {'Δ(B-G)':<8}")
    print("-" * 88)
    for prog_name, res in results.items():
        delta_cg = res["curiosity_final"] - res["greedy_final"]
        blind_final = res.get("blind_final", 0)
        delta_bg = blind_final - res["greedy_final"]
        print(f"{prog_name:<22} {res['corridor_depth']:<6} "
              f"{res['random_final']:<8.0f} {res['greedy_final']:<8.0f} "
              f"{res['curiosity_final']:<10.0f} {blind_final:<8.0f} "
              f"{delta_cg:<+8.0f} {delta_bg:<+8.0f}")


def print_coverage_curves(results: dict):
    """Print coverage curves in text format."""
    for prog_name, res in results.items():
        print(f"\n{prog_name}:")
        for strategy, key in [("random", "random_curve"),
                               ("greedy", "greedy_curve"),
                               ("curiosity", "curiosity_curve"),
                               ("blind", "blind_curve")]:
            curve = res.get(key, [])
            if curve:
                curve_str = " ".join(f"{v:3.0f}" for v in curve)
                print(f"  {strategy:>10}: {curve_str}")


def print_calibration_summary(results: dict):
    """Print calibration metrics summary."""
    print(f"\n{'Program':<22} {'Mode':<8} {'ρ(H,gain)':<12} "
          f"{'High-H gain':<13} {'Low-H gain':<13}")
    print("-" * 68)
    for prog_name, res in results.items():
        for mode, cal_key in [("visible", "calibration"), ("blind", "blind_calibration")]:
            cal = res.get(cal_key, {})
            if "spearman_correlation" in cal:
                print(f"{prog_name:<22} {mode:<8} {cal['spearman_correlation']:<12.3f} "
                      f"{cal['high_entropy_mean_gain']:<13.3f} "
                      f"{cal['low_entropy_mean_gain']:<13.3f}")
            elif cal:
                print(f"{prog_name:<22} {mode:<8} {'N/A':<12} {'N/A':<13} {'N/A':<13}")
