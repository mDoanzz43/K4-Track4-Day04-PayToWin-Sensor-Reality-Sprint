#!/usr/bin/env python3
"""
scripts/run_benchmark.py -- Main benchmark runner for LiDAR corruption lab.

Usage (from project root):
    # Single sample smoke test (sample 000000 only):
    python scripts/run_benchmark.py --smoke

    # Full run (10 baseline + 300 corrupted = 310 cases):
    python scripts/run_benchmark.py

    # Custom config:
    python scripts/run_benchmark.py --config configs/benchmark.json

Contract with Person 2 (src/metrics.py):
    evaluate(clean, corrupted, kind, range_bins) -> list[dict]
    Each dict has: metric, range_min_m, range_max_m, value, unit
    Runner adds: sample_id, corruption, severity, seed

Outputs:
    outputs/metrics.csv       -- all rows, schema per guild_project.md
    outputs/run_metadata.json -- provenance, commit, checksums, config, versions
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_io import load_kitti_bin, file_md5
from src.corruption_adapter import apply_corruption

# Config defaults
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "benchmark.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")
METADATA_JSON = os.path.join(OUTPUT_DIR, "run_metadata.json")

CSV_FIELDNAMES = [
    "sample_id", "corruption", "severity", "seed",
    "metric", "range_min_m", "range_max_m", "value", "unit"
]


# ---------------------------------------------------------------------------
# Metrics module integration (Person 2's interface)
# ---------------------------------------------------------------------------

def _try_import_metrics():
    """Attempt to import src.metrics from Person 2.
    Returns the module or None if not available."""
    try:
        import importlib
        metrics_mod = importlib.import_module("src.metrics")
        return metrics_mod
    except ImportError:
        return None


def _stub_evaluate(clean, corrupted, kind, range_bins):
    """
    STUB: used ONLY when src/metrics.py is not yet available from Person 2.
    This is a placeholder so the runner can run end-to-end for integration testing.
    It does NOT compute real metrics -- Person 2's implementation takes precedence.
    Records produced here are marked with unit='STUB_PENDING_PERSON2'.
    """
    records = []
    # Range bins: pairs of (lo, hi)
    bin_edges = list(zip(range_bins[:-1], range_bins[1:]))

    def range_mask(pts, lo, hi):
        r = np.sqrt((pts[:, :3] ** 2).sum(axis=1))
        return (r >= lo) & (r < hi)

    for (lo, hi) in bin_edges:
        clean_mask = range_mask(clean, lo, hi)
        corrupted_mask = range_mask(corrupted, lo, hi)
        n_clean = int(clean_mask.sum())
        n_corrupted = int(corrupted_mask.sum())
        records.append({
            "metric": "point_count",
            "range_min_m": lo,
            "range_max_m": hi,
            "value": n_corrupted,
            "unit": "STUB_PENDING_PERSON2",
        })
        if kind == "density_decrease":
            if n_clean > 0:
                ret = 100.0 * n_corrupted / n_clean
            else:
                ret = None
            records.append({
                "metric": "retention_percent",
                "range_min_m": lo,
                "range_max_m": hi,
                "value": ret,
                "unit": "STUB_PENDING_PERSON2",
            })
        if kind == "gaussian_noise":
            # Gaussian preserves shape/order; RMSE per-point XYZ
            if clean.shape == corrupted.shape:
                diff = corrupted[:, :3] - clean[:, :3]
                per_pt_sq = (diff ** 2).sum(axis=1)
                mask = range_mask(clean, lo, hi)
                if mask.sum() > 0:
                    rmse = float(np.sqrt(per_pt_sq[mask].mean()))
                else:
                    rmse = None
                records.append({
                    "metric": "xyz_rmse_m",
                    "range_min_m": lo,
                    "range_max_m": hi,
                    "value": rmse,
                    "unit": "STUB_PENDING_PERSON2",
                })
    return records


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_head(repo_path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config_path=CONFIG_PATH, smoke=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_root = os.path.join(os.path.dirname(__file__), "..", cfg["data_root"])
    velodyne_dir = os.path.join(data_root, "velodyne")
    range_bins = cfg["range_bins"]
    severities = cfg["severities"]
    seeds = cfg["seeds"]
    corruptions = cfg["corruptions"]

    # Discover sample IDs
    sample_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(velodyne_dir)
        if f.endswith(".bin")
    )
    if smoke:
        sample_ids = sample_ids[:1]
        print(f"[SMOKE] Running single sample: {sample_ids[0]}")

    # Try to import Person 2's metrics module and probe it
    metrics_mod = _try_import_metrics()
    _person2_metrics_ready = False
    if metrics_mod is not None:
        # Probe: call with tiny arrays to see if NotImplementedError stub
        try:
            import numpy as _np
            _dummy = _np.zeros((5, 4), dtype='float32')
            metrics_mod.evaluate(_dummy, _dummy, 'baseline', [0, 10, 80])
            _person2_metrics_ready = True
        except NotImplementedError:
            pass
        except Exception:
            # Other errors may mean real implementation with edge-case; treat as ready
            _person2_metrics_ready = True
    if _person2_metrics_ready:
        evaluate_fn = metrics_mod.evaluate
        print("[INFO] Using src.metrics.evaluate from Person 2")
    else:
        evaluate_fn = _stub_evaluate
        if metrics_mod is not None:
            print("[WARN] src/metrics.py exists but raises NotImplementedError -- "
                  "using internal STUB. Person 2: implement evaluate() to activate.")
        else:
            print("[WARN] src/metrics.py not found -- using STUB evaluate. "
                  "Replace with Person 2's implementation.")

    # Collect baseline checksums for provenance
    checksums = {}
    for sid in sample_ids:
        path = os.path.join(velodyne_dir, sid + ".bin")
        checksums[sid] = file_md5(path)

    # Run metadata init
    run_start = datetime.now(timezone.utc).isoformat()
    repo_commit = _git_head(os.path.join(os.path.dirname(__file__), "..", "3D_Corruptions_AD"))

    all_rows = []
    n_baseline = 0
    n_corrupted = 0
    errors = []

    total_cases = len(sample_ids) + len(sample_ids) * len(corruptions) * len(severities) * len(seeds)
    done = 0

    for sid in sample_ids:
        path = os.path.join(velodyne_dir, sid + ".bin")
        try:
            clean = load_kitti_bin(path)
        except Exception as e:
            errors.append({"sample_id": sid, "step": "load", "error": str(e)})
            print(f"[ERROR] {sid}: load failed -- {e}")
            continue

        # --- BASELINE (severity=0; NOT passed to upstream) ---
        try:
            baseline_records = evaluate_fn(clean, clean, "baseline", range_bins)
            for rec in baseline_records:
                all_rows.append({
                    "sample_id": sid,
                    "corruption": "baseline",
                    "severity": 0,
                    "seed": "",
                    **rec,
                })
            n_baseline += 1
        except Exception as e:
            errors.append({"sample_id": sid, "step": "baseline_eval", "error": str(e)})
            print(f"[ERROR] {sid} baseline eval -- {e}")
        done += 1
        if done % 5 == 0 or smoke:
            print(f"  Progress: {done}/{total_cases} cases")

        # --- CORRUPTED ---
        for kind in corruptions:
            for sev in severities:
                for seed in seeds:
                    try:
                        corrupted = apply_corruption(clean, kind, sev, seed)
                        # Verify baseline not mutated
                        orig_copy = load_kitti_bin(path)
                        if not np.array_equal(clean, orig_copy):
                            errors.append({
                                "sample_id": sid, "kind": kind, "sev": sev, "seed": seed,
                                "step": "mutation_check", "error": "clean was mutated!"
                            })
                        records = evaluate_fn(clean, corrupted, kind, range_bins)
                        for rec in records:
                            all_rows.append({
                                "sample_id": sid,
                                "corruption": kind,
                                "severity": sev,
                                "seed": seed,
                                **rec,
                            })
                        n_corrupted += 1
                    except Exception as e:
                        errors.append({
                            "sample_id": sid, "kind": kind, "sev": sev, "seed": seed,
                            "step": "corrupt_eval", "error": str(e)
                        })
                        print(f"[ERROR] {sid}/{kind}/sev{sev}/seed{seed} -- {e}")
                    done += 1
                    if done % 30 == 0:
                        print(f"  Progress: {done}/{total_cases} cases")

    # Write metrics CSV
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[DONE] metrics.csv: {len(all_rows)} rows -> {METRICS_CSV}")

    # Write metadata JSON
    run_end = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_start_utc": run_start,
        "run_end_utc": run_end,
        "command": " ".join(sys.argv),
        "smoke_mode": smoke,
        "config_path": os.path.abspath(config_path),
        "config": cfg,
        "upstream": {
            "url": "https://github.com/thu-ml/3D_Corruptions_AD",
            "local_path": "3D_Corruptions_AD",
            "commit": repo_commit,
            "license": "MIT -- Copyright (c) 2023 Tsinghua Machine Learning Group",
        },
        "adaptation_notes": {
            "gaussian_noise": "XYZ only; intensity column not perturbed (upstream adds noise to all C cols)",
            "density_dec_global": "Verbatim from upstream; drops 6/12/18/24/30% of points",
            "severity_0": "NOT passed to upstream functions; baseline uses clean cloud",
        },
        "baseline_checksums_md5": checksums,
        "sample_ids": sample_ids,
        "n_baseline": n_baseline,
        "n_corrupted": n_corrupted,
        "errors": errors,
        "metrics_csv": os.path.abspath(METRICS_CSV),
        "units": {
            "point_count": "points/bin",
            "retention_percent": "percent",
            "xyz_rmse_m": "metres",
        },
        "range_bins_m": range_bins,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "evaluate_source": "src.metrics" if metrics_mod is not None else "STUB_run_benchmark",
        "unverified_assumptions": [
            "velodyne_reduced origin not confirmed -- excluded from benchmark",
            "Dataset assumed KITTI but source URL not on-file; verified empirically",
        ],
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[DONE] run_metadata.json -> {METADATA_JSON}")

    if errors:
        print(f"[WARN] {len(errors)} errors encountered (see run_metadata.json)")
    print(f"Summary: {n_baseline} baseline, {n_corrupted} corrupted cases processed")

    return 0 if not errors else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR corruption benchmark runner")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to benchmark.json")
    parser.add_argument("--smoke", action="store_true",
                        help="Run single sample end-to-end (fast check)")
    args = parser.parse_args()
    sys.exit(run(config_path=args.config, smoke=args.smoke))