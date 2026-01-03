#!/usr/bin/env python3
"""
Compute summary stats and paired tests from quality outputs.
"""

from pathlib import Path
import argparse
import json

import numpy as np
from scipy import stats


def load_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "detailed_results" in data:
        return data
    if "strategies" in data and data["strategies"]:
        return data["strategies"][0]
    raise ValueError("No results found in JSON.")


def paired_stats(baseline, intervened):
    baseline = np.array(baseline, dtype=float)
    intervened = np.array(intervened, dtype=float)
    diff = baseline - intervened
    t_stat, p_value = stats.ttest_rel(baseline, intervened)
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
    return {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/activation_patching/quality_assessment.json",
        help="Path to quality assessment JSON",
    )
    parser.add_argument(
        "--output",
        default="results/activation_patching/quality_stats.json",
        help="Path to write stats JSON",
    )
    args = parser.parse_args()

    result = load_results(args.input)
    results = result["detailed_results"]

    baseline_tokens = [r["baseline"]["tokens"] for r in results]
    intervened_tokens = [r["intervened"]["tokens"] for r in results]
    baseline_markers = [r["baseline"]["reasoning_markers"] for r in results]
    intervened_markers = [r["intervened"]["reasoning_markers"] for r in results]

    token_stats = paired_stats(baseline_tokens, intervened_tokens)
    marker_stats = paired_stats(baseline_markers, intervened_markers)

    out = {
        "input": str(args.input),
        "token_stats": token_stats,
        "marker_stats": marker_stats,
    }

    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Token reduction (baseline - intervened):")
    print(
        f"  n={token_stats['n']} mean={token_stats['mean_diff']:.2f} std={token_stats['std_diff']:.2f} "
        f"t={token_stats['t_stat']:.2f} p={token_stats['p_value']:.4g} d={token_stats['cohens_d']:.2f}"
    )
    print("Marker reduction (baseline - intervened):")
    print(
        f"  n={marker_stats['n']} mean={marker_stats['mean_diff']:.2f} std={marker_stats['std_diff']:.2f} "
        f"t={marker_stats['t_stat']:.2f} p={marker_stats['p_value']:.4g} d={marker_stats['cohens_d']:.2f}"
    )


if __name__ == "__main__":
    main()
