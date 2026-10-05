"""Headless plots for benchmark evidence.

The three lexicographically first scans are selected to avoid cherry-picking;
severity 5 and seed 42 are used for the BEV comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.corruption_adapter import apply_corruption
from src.data_io import load_kitti_bin

SIGMA_BY_SEVERITY = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.10}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(text):
    return None if text is None or text == "" else float(text)


def _range_label(row):
    lo, hi = row["range_min_m"], row["range_max_m"]
    if lo == "" and hi == "":
        return "global"
    if hi == "":
        return f"[{float(lo):g}, +inf) m"
    return f"[{float(lo):g}, {float(hi):g}) m"


def plot_bev(data_dir, output_dir, sample_ids, seed=42, severity=5,
             display_limit=50000):
    for sample_id in sample_ids:
        clean = load_kitti_bin(os.path.join(data_dir, sample_id + ".bin"))
        density = apply_corruption(clean, "density_decrease", severity, seed)
        gaussian = apply_corruption(clean, "gaussian_noise", severity, seed)
        clouds = (("Baseline", clean), ("Density severity 5", density),
                  ("Gaussian severity 5", gaussian))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, (title, points) in zip(axes, clouds):
            # Deterministic stride sampling affects rendering only, never metrics.
            stride = max(1, int(np.ceil(len(points) / display_limit)))
            shown = points[::stride]
            ax.scatter(shown[:, 0], shown[:, 1], s=0.15, c=shown[:, 3],
                       cmap="viridis", vmin=0, vmax=1, rasterized=True)
            ax.set(title=title, xlabel="x (m)", ylabel="y (m)",
                   xlim=(-80, 80), ylim=(-60, 60), aspect="equal")
            ax.grid(alpha=0.15)
        fig.suptitle(f"BEV sample {sample_id} - seed {seed}; deterministic display subsampling")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"bev_{sample_id}.png"), dpi=180)
        plt.close(fig)


def plot_retention(summary_rows, output_dir):
    rows = [r for r in summary_rows
            if r["corruption"] == "density_decrease"
            and r["metric"] == "retention_percent"
            and not (r["range_min_m"] == "" and r["range_max_m"] == "")]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sorted({_range_label(r) for r in rows})
    for label in labels:
        selected = sorted((r for r in rows if _range_label(r) == label),
                          key=lambda r: int(r["severity"]))
        selected = [r for r in selected if _number(r["mean_across_samples"]) is not None]
        if not selected:
            continue
        x = [int(r["severity"]) for r in selected]
        y = [_number(r["mean_across_samples"]) for r in selected]
        err = [_number(r["std_across_samples"]) or 0 for r in selected]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=label)
    ax.set(xlabel="Severity", ylabel="Mean retention (%)", xticks=range(1, 6),
           title="Density retention by clean range bin (error bars: between-scene SD)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "retention_by_severity_range.png"), dpi=180)
    plt.close(fig)


def plot_rmse(summary_rows, output_dir):
    rows = [r for r in summary_rows
            if r["corruption"] == "gaussian_noise" and r["metric"] == "xyz_rmse_m"
            and r["range_min_m"] == "" and r["range_max_m"] == ""]
    rows.sort(key=lambda r: int(r["severity"]))
    x = [SIGMA_BY_SEVERITY[int(r["severity"])] for r in rows]
    y = [_number(r["mean_across_samples"]) for r in rows]
    err = [_number(r["std_across_samples"]) or 0 for r in rows]
    expected = [np.sqrt(3) * sigma for sigma in x]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label="Measured")
    ax.plot(x, expected, "--", label="Expected sqrt(3) * sigma")
    ax.set(xlabel="Gaussian coordinate sigma (m)", ylabel="Global XYZ RMSE (m)",
           title="Gaussian displacement RMSE (error bars: between-scene SD)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rmse_by_sigma.png"), dpi=180)
    plt.close(fig)


def generate(data_dir=os.path.join(ROOT, "data", "velodyne"),
             summary_path=os.path.join(ROOT, "outputs", "summary_overall.csv"),
             output_dir=os.path.join(ROOT, "outputs", "figures"), samples=3):
    os.makedirs(output_dir, exist_ok=True)
    sample_ids = sorted(os.path.splitext(name)[0] for name in os.listdir(data_dir)
                        if name.endswith(".bin"))[:samples]
    if len(sample_ids) < 3:
        raise ValueError("At least three .bin scans are required for BEV evidence")
    rows = _read_csv(summary_path)
    plot_bev(data_dir, output_dir, sample_ids)
    plot_retention(rows, output_dir)
    plot_rmse(rows, output_dir)
    print(f"[DONE] figures for samples {', '.join(sample_ids)} -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "velodyne"))
    parser.add_argument("--summary", default=os.path.join(ROOT, "outputs", "summary_overall.csv"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs", "figures"))
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    generate(args.data_dir, args.summary, args.output_dir, args.samples)
