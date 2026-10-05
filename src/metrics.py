"""
src/metrics.py -- CONTRACT FILE for Person 2.

Person 2 owns and implements this file.
Person 1 runner (scripts/run_benchmark.py) imports:
    from src.metrics import evaluate

CONTRACT:
    evaluate(clean, corrupted, kind, range_bins) -> list[dict]

    Parameters
    ----------
    clean      : np.ndarray float32 (N, 4)  -- baseline point cloud
    corrupted  : np.ndarray float32 (N', 4) -- corrupted version
    kind       : str  -- "baseline", "density_decrease", or "gaussian_noise"
    range_bins : list[float]  -- e.g. [0, 10, 20, 40, 80]

    Returns
    -------
    list of dicts, each with keys:
        metric       : str  -- "point_count", "retention_percent", or "xyz_rmse_m"
        range_min_m  : float or None  -- bin lower bound (None for global)
        range_max_m  : float or None  -- bin upper bound (None for global)
        value        : float or None  -- None for empty-bin NA
        unit         : str  -- "points/bin", "percent", "metres"

    Runner adds: sample_id, corruption, severity, seed

CSV schema (guild_project.md):
    sample_id, corruption, severity, seed,
    metric, range_min_m, range_max_m, value, unit

Notes for Person 2:
- range: r = sqrt(x^2 + y^2 + z^2) in metres (KITTI velodyne frame)
- Bins: [0,10), [10,20), [20,40), [40,80]; report out-of-range separately
- retention_percent = 100 * N_corrupted_bin / N_clean_bin
  baseline empty bin -> NA (None), NOT 0
- xyz_rmse_m = sqrt(mean(sum((corrupted_xyz - clean_xyz)^2, axis=1)))
  ONLY valid when clean.shape == corrupted.shape (gaussian_noise contract)
  Do NOT compute RMSE by row-matching after dropout
- Baseline RMSE = 0.0
- unit for point_count is "points/bin" (not points/m^3)
- Gaussian may shift points across range bins; retention metric not meaningful for gaussian
"""

import numpy as np


def evaluate(clean, corrupted, kind, range_bins):
    """
    PLACEHOLDER -- to be implemented by Person 2.

    This stub raises NotImplementedError so Person 1's runner will fall back
    to the internal STUB in run_benchmark.py during development.
    Remove the raise and implement below.
    """
    raise NotImplementedError(
        "src/metrics.py not yet implemented by Person 2. "
        "Runner will use internal stub until this is available."
    )

    # --- Person 2 implements below this line ---
    # records = []
    # ...
    # return records