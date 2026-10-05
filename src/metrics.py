"""Sensor-level metrics for the LiDAR corruption benchmark.

Range bins are left-closed and right-open. Because Euclidean range is
non-negative, the extra out-of-range bin is ``[range_bins[-1], +inf)`` and is
represented with a blank upper bound in CSV. Rows with both bounds blank are
global metrics.
"""

from __future__ import annotations

import numpy as np


VALID_KINDS = ("baseline", "density_decrease", "gaussian_noise")


def _validate_points(name, points):
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_bins(range_bins):
    edges = np.asarray(range_bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("range_bins must contain at least two ordered edges")
    if not np.all(np.isfinite(edges)) or edges[0] < 0:
        raise ValueError("range_bins must be finite and non-negative")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("range_bins must be strictly increasing")
    return edges


def _ranges(points):
    xyz = points[:, :3].astype(np.float64, copy=False)
    return np.sqrt(np.einsum("ij,ij->i", xyz, xyz))


def _record(metric, lo, hi, value, unit):
    return {
        "metric": metric,
        "range_min_m": None if lo is None else float(lo),
        "range_max_m": None if hi is None else float(hi),
        "value": None if value is None else float(value),
        "unit": unit,
    }


def evaluate(clean, corrupted, kind, range_bins):
    """Return metric records for one clean/corrupted point-cloud pair.

    Point counts are emitted per range bin. Density additionally emits per-bin
    and global retention. Gaussian emits corresponding-point XYZ RMSE. Baseline
    emits zero RMSE as an explicit reference. Empty clean bins return ``None``
    for ratio/RMSE metrics and are never replaced with zero.
    """
    clean = _validate_points("clean", clean)
    corrupted = _validate_points("corrupted", corrupted)
    edges = _validate_bins(range_bins)
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
    if kind in ("baseline", "gaussian_noise") and clean.shape != corrupted.shape:
        raise ValueError(
            f"{kind} requires matching clean/corrupted shapes for row-wise RMSE: "
            f"{clean.shape} != {corrupted.shape}"
        )

    clean_r = _ranges(clean)
    corrupted_r = _ranges(corrupted)
    bins = [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    bins.append((float(edges[-1]), None))
    records = []

    if kind in ("baseline", "gaussian_noise"):
        delta_sq = np.sum(
            (corrupted[:, :3].astype(np.float64) - clean[:, :3].astype(np.float64)) ** 2,
            axis=1,
        )
    else:
        delta_sq = None

    for lo, hi in bins:
        if hi is None:
            clean_mask = clean_r >= lo
            corrupted_mask = corrupted_r >= lo
        else:
            clean_mask = (clean_r >= lo) & (clean_r < hi)
            corrupted_mask = (corrupted_r >= lo) & (corrupted_r < hi)

        n_clean = int(np.count_nonzero(clean_mask))
        n_corrupted = int(np.count_nonzero(corrupted_mask))
        records.append(_record("point_count", lo, hi, n_corrupted, "points/bin"))

        if kind == "density_decrease":
            retention = 100.0 * n_corrupted / n_clean if n_clean else None
            records.append(_record("retention_percent", lo, hi, retention, "percent"))
        elif kind in ("baseline", "gaussian_noise"):
            rmse = float(np.sqrt(np.mean(delta_sq[clean_mask]))) if n_clean else None
            records.append(_record("xyz_rmse_m", lo, hi, rmse, "metres"))

    if kind == "density_decrease":
        global_retention = 100.0 * len(corrupted) / len(clean) if len(clean) else None
        records.append(_record("retention_percent", None, None, global_retention, "percent"))
    else:
        global_rmse = float(np.sqrt(np.mean(delta_sq))) if len(clean) else None
        records.append(_record("xyz_rmse_m", None, None, global_rmse, "metres"))

    return records


def _self_test():
    clean = np.array(
        [[0, 0, 0, 1], [10, 0, 0, 1], [79, 0, 0, 1], [80, 0, 0, 1]],
        dtype=np.float32,
    )
    baseline = evaluate(clean, clean, "baseline", [0, 10, 20, 40, 80])
    counts = [r["value"] for r in baseline if r["metric"] == "point_count"]
    assert counts == [1.0, 1.0, 0.0, 1.0, 1.0], counts
    moved = clean.copy()
    moved[:, 1] += 1
    gaussian = evaluate(clean, moved, "gaussian_noise", [0, 10, 20, 40, 80])
    global_rmse = [
        r["value"] for r in gaussian
        if r["metric"] == "xyz_rmse_m" and r["range_min_m"] is None
    ][0]
    assert np.isclose(global_rmse, 1.0), global_rmse
    empty = evaluate(clean[:1], clean[:0], "density_decrease", [0, 1, 2])
    assert any(r["value"] is None for r in empty if r["metric"] == "retention_percent")
    print("[OK] metrics self-test passed")


if __name__ == "__main__":
    _self_test()
