"""
src/data_io.py -- KITTI LiDAR point cloud loader

ASSUMPTIONS / UNVERIFIED:
- Dataset is KITTI-format: binary float32 little-endian, N x 4 (x, y, z, intensity).
  Verified empirically on 000000.bin: file_size % 16 == 0, x/y/z in metres,
  intensity in [0, 1], all finite. See outputs/manifest.csv for per-sample check.
- data/velodyne_reduced/ is NOT used as a corruption source.
  UNVERIFIED: exact origin of reduced files (could be camera-FOV crop or other pipeline).
  Ratio velodyne_reduced/velodyne is approx 17-18% -- different from density_dec_global drops.
  Do NOT compare reduced vs velodyne as a severity level until origin is confirmed.
- Calibration follows KITTI format (P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo).
  Parsed but NOT used in MVP benchmark (LiDAR-only pipeline).
- Labels follow KITTI object label format (15 fields per object).
  NOT used in MVP; not validated for coordinate frame.
"""

import os
import hashlib
import numpy as np


def load_kitti_bin(path):
    """Load KITTI velodyne binary as float32 (N, 4) array [x, y, z, intensity].

    Validates:
    - File exists and is non-empty
    - File size divisible by 16 (4 x float32)
    - All values finite
    - At least 1 point

    Returns float32 (N, 4) ndarray.
    Raises ValueError with descriptive message if validation fails.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Point cloud not found: {}".format(path))

    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Empty file: {}".format(path))
    if size % 16 != 0:
        raise ValueError(
            "File size {} not divisible by 16 -- not KITTI float32 N x 4: {}".format(size, path)
        )

    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)

    if points.shape[0] == 0:
        raise ValueError("Zero points read from: {}".format(path))
    if not np.all(np.isfinite(points)):
        n_bad = int(np.sum(~np.isfinite(points)))
        raise ValueError("{} non-finite values in: {}".format(n_bad, path))

    return points


def file_md5(path):
    """Return hex MD5 of file (for provenance metadata)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_root):
    """
    Scan 5 data subdirectories, collect per-ID file info, report missing IDs.

    Checks: calib/*.txt, image/*.png, label/*.txt, velodyne/*.bin,
            velodyne_reduced/*.bin

    Returns list of dicts, one per (folder, sample_id).
    Prints warnings for IDs present in some folders but missing in others.
    """
    subdirs = {
        "calib": ".txt",
        "image": ".png",
        "label": ".txt",
        "velodyne": ".bin",
        "velodyne_reduced": ".bin",
    }

    present = {}
    for folder, ext in subdirs.items():
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            print("[WARN] Folder not found: {}".format(folder_path))
            present[folder] = set()
            continue
        stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(folder_path)
            if f.endswith(ext)
        }
        present[folder] = stems

    all_ids = sorted(set().union(*present.values()))
    reference_ids = present.get("velodyne", set())

    records = []
    for sample_id in all_ids:
        for folder, ext in subdirs.items():
            folder_path = os.path.join(data_root, folder)
            fpath = os.path.join(folder_path, sample_id + ext)
            exists = sample_id in present.get(folder, set())
            size = os.path.getsize(fpath) if exists else None
            md5 = file_md5(fpath) if (exists and folder == "velodyne") else None
            records.append(
                {
                    "sample_id": sample_id,
                    "folder": folder,
                    "filename": sample_id + ext,
                    "exists": exists,
                    "size_bytes": size,
                    "md5": md5,
                }
            )

    for folder in subdirs:
        missing = reference_ids - present.get(folder, set())
        extra = present.get(folder, set()) - reference_ids
        if missing:
            print("[MISSING] {}: IDs in velodyne but not here -> {}".format(folder, sorted(missing)))
        if extra:
            print("[EXTRA]   {}: IDs here but not in velodyne -> {}".format(folder, sorted(extra)))

    return records


def describe_bin(path):
    """Return dict of basic stats for a single .bin file."""
    pts = load_kitti_bin(path)
    xyz = pts[:, :3]
    r = np.sqrt((xyz ** 2).sum(axis=1))
    return {
        "n_points": int(pts.shape[0]),
        "x_min": float(xyz[:, 0].min()),
        "x_max": float(xyz[:, 0].max()),
        "y_min": float(xyz[:, 1].min()),
        "y_max": float(xyz[:, 1].max()),
        "z_min": float(xyz[:, 2].min()),
        "z_max": float(xyz[:, 2].max()),
        "intensity_min": float(pts[:, 3].min()),
        "intensity_max": float(pts[:, 3].max()),
        "range_min_m": float(r.min()),
        "range_max_m": float(r.max()),
        "all_finite": bool(np.all(np.isfinite(pts))),
    }