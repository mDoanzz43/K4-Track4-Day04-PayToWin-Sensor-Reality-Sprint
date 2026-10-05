#!/usr/bin/env python3
"""
scripts/check_dataset.py -- Dataset verification and manifest generation.

Usage:
    python scripts/check_dataset.py

Outputs:
    outputs/manifest.csv  -- per-(folder, sample_id) existence and size
    Prints stats for each velodyne scan.
"""

import os
import sys
import csv

# Allow importing src/ from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_io import build_manifest, describe_bin

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MANIFEST_CSV = os.path.join(OUTPUT_DIR, "manifest.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Building manifest for 5 data folders")
    print("=" * 60)

    records = build_manifest(DATA_ROOT)

    # Write manifest CSV
    fieldnames = ["sample_id", "folder", "filename", "exists", "size_bytes", "md5"]
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Manifest written: {MANIFEST_CSV}")

    # Summary
    velodyne_records = [r for r in records if r["folder"] == "velodyne"]
    present = [r for r in velodyne_records if r["exists"]]
    missing = [r for r in velodyne_records if not r["exists"]]
    print(f"velodyne: {len(present)} present, {len(missing)} missing")
    if missing:
        print(f"  Missing IDs: {[r['sample_id'] for r in missing]}")

    print()
    print("=" * 60)
    print("STEP 2: Binary schema verification (velodyne/*.bin)")
    print("=" * 60)
    print("Expected: float32 little-endian, N x 4, XYZ in metres, intensity in [0,1]")
    print()

    velodyne_dir = os.path.join(DATA_ROOT, "velodyne")
    all_ok = True
    for r in sorted(velodyne_records, key=lambda x: x["sample_id"]):
        if not r["exists"]:
            print(f"  {r['sample_id']}: MISSING -- skip")
            all_ok = False
            continue
        path = os.path.join(velodyne_dir, r["filename"])
        try:
            stats = describe_bin(path)
            size_bytes = r["size_bytes"]
            mod16 = size_bytes % 16
            ok_str = "OK" if mod16 == 0 else f"WARN size%16={mod16}"
            print(
                f"  {r['sample_id']}: {stats['n_points']:>7} pts | "
                f"x[{stats['x_min']:+.1f},{stats['x_max']:+.1f}] "
                f"y[{stats['y_min']:+.1f},{stats['y_max']:+.1f}] "
                f"z[{stats['z_min']:+.1f},{stats['z_max']:+.1f}] "
                f"int[{stats['intensity_min']:.2f},{stats['intensity_max']:.2f}] "
                f"range[{stats['range_min_m']:.1f},{stats['range_max_m']:.1f}]m "
                f"finite={stats['all_finite']} {ok_str}"
            )
        except Exception as e:
            print(f"  {r['sample_id']}: ERROR -- {e}")
            all_ok = False

    print()
    print("=" * 60)
    print("STEP 3: velodyne_reduced size ratio (NOT used as corruption)")
    print("=" * 60)
    print("NOTE: Origin of velodyne_reduced is UNVERIFIED.")
    print("      Possibly camera-FOV crop. NOT compared as severity level.")
    print()
    reduced_dir = os.path.join(DATA_ROOT, "velodyne_reduced")
    for r in sorted(velodyne_records, key=lambda x: x["sample_id"]):
        if not r["exists"]:
            continue
        red_path = os.path.join(reduced_dir, r["filename"])
        vel_path = os.path.join(velodyne_dir, r["filename"])
        if os.path.isfile(red_path):
            red_size = os.path.getsize(red_path)
            vel_size = r["size_bytes"]
            ratio = red_size / vel_size if vel_size else float("nan")
            print(f"  {r['sample_id']}: reduced/velodyne size ratio = {ratio:.3f} ({red_size} / {vel_size} bytes)")
        else:
            print(f"  {r['sample_id']}: velodyne_reduced missing")

    print()
    status = "PASS" if all_ok else "FAIL (see above)"
    print(f"Dataset check result: {status}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())