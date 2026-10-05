"""
src/corruption_adapter.py -- Adapter wrapping two functions from 3D_Corruptions_AD.

UPSTREAM SOURCE
  Repository : https://github.com/thu-ml/3D_Corruptions_AD
  License    : MIT (Copyright (c) 2023 Tsinghua Machine Learning Group)
  Commit     : 48c23f77fe82beab599f8248b7794928334a3fb5
  File       : LiDAR_corruptions.py

FUNCTIONS EXTRACTED (minimal copy; NO other functions included)
  - density_dec_global
  - gaussian_noise  (adapted -- see ADAPTATION note below)

WHY NOT DIRECT IMPORT:
  LiDAR_corruptions.py has top-level imports of open3d, h5py, and distortion
  which are heavy / unavailable in this environment. The two functions we need
  use only numpy.  We extract them verbatim from the commit above, keeping the
  MIT license header and attribution comment.

CHANGES vs UPSTREAM (density_dec_global):
  - None. Function is identical to upstream.

CHANGES vs UPSTREAM (gaussian_noise):
  ADAPTATION: Upstream adds jitter to ALL C columns (including intensity when C=4).
  Our adapter applies noise ONLY to XYZ columns ([:, :3]) and leaves intensity
  column ([:, 3]) unchanged. This is required by the MVP contract:
    "Gaussian: giuu intensity, so diem va thu tu"
  The sigma values [0.02, 0.04, 0.06, 0.08, 0.10] per severity are unchanged.
  The upstream comment says "N*3" suggesting the original was designed for 3-col
  input; our adaptation makes the intent explicit for 4-col KITTI data.

CONTRACT: apply_corruption(points, kind, severity, seed) -> np.ndarray
  - points  : float32 (N, 4) -- NOT mutated
  - kind    : "density_decrease" | "gaussian_noise"
  - severity: int 1-5
  - seed    : int (for reproducibility)
  - returns : float32 (N_out, 4)
              density_decrease: N_out <= N, subset of original rows, intensity kept
              gaussian_noise  : N_out == N, same order, only XYZ perturbed
"""

import numpy as np


# ---------------------------------------------------------------------------
# Extracted from LiDAR_corruptions.py @ 48c23f7 -- MIT License
# Copyright (c) 2023 Tsinghua Machine Learning Group
# ---------------------------------------------------------------------------

def _density_dec_global(pointcloud, severity):
    """
    Source: density_dec_global in LiDAR_corruptions.py (verbatim).
    Drops 6/12/18/24/30% of points randomly (severity 1-5).
    Uses numpy random; seed must be set BEFORE calling.
    """
    N, C = pointcloud.shape
    num = int(N * 0.3)
    c = [int(0.2 * num), int(0.4 * num), int(0.6 * num), int(0.8 * num), num][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud


def _gaussian_noise_xyz_only(pointcloud, severity):
    """
    ADAPTED from gaussian_noise in LiDAR_corruptions.py @ 48c23f7.
    CHANGE: jitter applied only to columns [:, :3] (XYZ); intensity col untouched.
    Sigma: [0.02, 0.04, 0.06, 0.08, 0.10] metres per severity level.
    Point count and order are preserved (shape unchanged).
    """
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.normal(size=(N, 3)) * c  # XYZ only
    out = pointcloud.copy().astype("float32")
    out[:, :3] += jitter.astype("float32")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_KINDS = ("density_decrease", "gaussian_noise")


def apply_corruption(
    points: np.ndarray,
    kind: str,
    severity: int,
    seed: int,
) -> np.ndarray:
    """Apply a corruption to a point cloud without mutating the input.

    Parameters
    ----------
    points   : float32 (N, 4) -- x, y, z, intensity.  NOT mutated.
    kind     : "density_decrease" or "gaussian_noise"
    severity : int 1-5  (caller must NOT pass severity=0 here; use baseline branch)
    seed     : int for numpy RandomState

    Returns
    -------
    float32 (N_out, 4) -- see module docstring for shape guarantees.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
    if not (1 <= severity <= 5):
        raise ValueError(f"severity must be 1-5, got {severity}")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must be (N, 4), got shape {points.shape}")

    # Work on a copy so we never mutate caller's array
    pts = points.copy()

    rng_state = np.random.RandomState(seed)
    # Set the global numpy seed; upstream functions use np.random directly
    np.random.seed(rng_state.randint(0, 2**31 - 1))

    if kind == "density_decrease":
        result = _density_dec_global(pts, severity)
    elif kind == "gaussian_noise":
        result = _gaussian_noise_xyz_only(pts, severity)

    return result.astype("float32")


def smoke_check():
    """Quick self-test.  Run: python -m src.corruption_adapter"""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 4)).astype("float32")
    pts[:, 3] = 0.5  # fixed intensity

    # --- Baseline: input not mutated ---
    original = pts.copy()

    # Gaussian: shape, order, intensity preserved
    g_out = apply_corruption(pts, "gaussian_noise", severity=3, seed=42)
    assert g_out.shape == pts.shape, f"Gaussian shape mismatch: {g_out.shape}"
    assert np.allclose(g_out[:, 3], pts[:, 3]), "Gaussian mutated intensity"
    assert not np.allclose(g_out[:, :3], pts[:, :3]), "Gaussian made no change to XYZ"
    assert np.array_equal(pts, original), "apply_corruption mutated input (gaussian)"

    # Reproducibility
    g_out2 = apply_corruption(pts, "gaussian_noise", severity=3, seed=42)
    assert np.array_equal(g_out, g_out2), "Gaussian not reproducible with same seed"

    # Density: subset + intensity kept + point count matches upstream formula
    d_out = apply_corruption(pts, "density_decrease", severity=1, seed=42)
    N = 1000
    num = int(N * 0.3)
    expected_removed = int(0.2 * num)
    expected_N_out = N - expected_removed
    assert d_out.shape[0] == expected_N_out, (
        f"Density drop count wrong: got {d_out.shape[0]}, expected {expected_N_out}"
    )
    assert d_out.shape[1] == 4, "Density lost intensity column"
    assert np.array_equal(pts, original), "apply_corruption mutated input (density)"

    # All remaining rows are from original (subset, not new points)
    # Check that all rows of d_out exist in pts (float equality OK since no math applied)
    pts_set = set(map(tuple, pts.tolist()))
    for row in d_out:
        assert tuple(row.tolist()) in pts_set, "Density returned row not in original"

    print("[OK] smoke_check passed: shape, order, intensity, reproducibility, subset")


if __name__ == "__main__":
    smoke_check()