#!/usr/bin/env python3
"""Aggregate benchmark metrics without replacing missing values by zero."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT = os.path.join(ROOT, "outputs", "metrics.csv")
DEFAULT_PER_SAMPLE = os.path.join(ROOT, "outputs", "summary_per_sample.csv")
DEFAULT_OVERALL = os.path.join(ROOT, "outputs", "summary_overall.csv")

DIMENSIONS = ("corruption", "severity", "metric", "range_min_m", "range_max_m", "unit")
PER_SAMPLE_FIELDS = ("sample_id", *DIMENSIONS, "mean", "std_across_seeds", "n_valid_seeds")
OVERALL_FIELDS = (*DIMENSIONS, "mean_across_samples", "std_across_samples", "n_valid_samples")


def _parse_value(text):
    if text is None or not text.strip():
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _stats(values):
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if valid.size == 0:
        return None, None, 0
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size >= 2 else None
    return mean, std, int(valid.size)


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(input_path=DEFAULT_INPUT, per_sample_path=DEFAULT_PER_SAMPLE,
              overall_path=DEFAULT_OVERALL):
    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "seed", "value", *DIMENSIONS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metrics CSV missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["sample_id"], *(row[name] for name in DIMENSIONS))
        grouped[key].append(_parse_value(row["value"]))

    per_sample_rows = []
    for key in sorted(grouped):
        mean, std, n_valid = _stats(grouped[key])
        per_sample_rows.append(dict(zip(PER_SAMPLE_FIELDS, (*key, mean, std, n_valid))))

    overall_groups = defaultdict(list)
    for row in per_sample_rows:
        key = tuple(row[name] for name in DIMENSIONS)
        overall_groups[key].append(row["mean"])

    overall_rows = []
    for key in sorted(overall_groups):
        mean, std, n_valid = _stats(overall_groups[key])
        overall_rows.append(dict(zip(OVERALL_FIELDS, (*key, mean, std, n_valid))))

    _write_csv(per_sample_path, PER_SAMPLE_FIELDS, per_sample_rows)
    _write_csv(overall_path, OVERALL_FIELDS, overall_rows)
    print(f"[DONE] {len(per_sample_rows)} rows -> {per_sample_path}")
    print(f"[DONE] {len(overall_rows)} rows -> {overall_path}")
    return per_sample_rows, overall_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--per-sample", default=DEFAULT_PER_SAMPLE)
    parser.add_argument("--overall", default=DEFAULT_OVERALL)
    args = parser.parse_args()
    summarize(args.input, args.per_sample, args.overall)
