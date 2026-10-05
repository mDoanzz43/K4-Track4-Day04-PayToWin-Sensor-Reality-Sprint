# K4-Track4-Day04 — Lab T2 LiDAR Corruption Benchmark

**Nhom:** PayToWin

| STT | Thanh Vien | MSSV | Phan Cong |
|-----|------------|------|-----------|
| 1 | Đỗ Mạnh Đoan | 2A202602839 | Data, corruption, runner, config  |
| 2 | Nguyễn Mạnh Cường | 2A2026.... | |

---

## Trang Thai

| Phan | Nguoi | Trang Thai |
|------|-------|------------|
| `src/data_io.py` | Nguoi 1 | DONE |
| `src/corruption_adapter.py` | Nguoi 1 | DONE |
| `scripts/check_dataset.py` | Nguoi 1 | DONE |
| `scripts/run_benchmark.py` | Nguoi 1 | DONE |
| `configs/benchmark.json` | Nguoi 1 | DONE |
| `requirements.txt` | Nguoi 1 | DONE |
| `docs/source_notes.md` | Nguoi 1 | DONE |
| `outputs/manifest.csv` | Nguoi 1 | DONE |
| `outputs/metrics.csv` | Nguoi 1 (STUB) | Cho Nguoi 2 implement evaluate() |
| `outputs/run_metadata.json` | Nguoi 1 | DONE |
| `src/metrics.py` | **Nguoi 2** | PENDING — contract co san, dien implement |
| `src/visualization.py` | **Nguoi 2** | PENDING |
| `scripts/summarize_results.py` | **Nguoi 2** | PENDING |
| `outputs/summary_per_sample.csv` | **Nguoi 2** | PENDING |
| `outputs/summary_overall.csv` | **Nguoi 2** | PENDING |
| `outputs/figures/` | **Nguoi 2** | PENDING |
| `docs/report.md` | **Nguoi 2** | PENDING |

---

## Cai dat moi truong

```bash
pip install -r requirements.txt
```

**Yeu cau:** Python >= 3.9, numpy >= 1.24. Khong can open3d, h5py, hay model checkpoint.

---

## Ket qua Nguoi 1 da ban giao

### Kiem tra dataset

```bash
python scripts/check_dataset.py
```

Ket qua da chay:
- 10/10 sample ID khop day du giua 5 thu muc (calib, image, label, velodyne, velodyne_reduced)
- Schema binary KITTI xac nhan: float32 little-endian N x 4, XYZ don vi met, intensity [0,1], all finite
- Manifest luu tai `outputs/manifest.csv`

### Chay benchmark (co the tai tao)

```bash
# Smoke 1 sample (kiem tra nhanh)
python scripts/run_benchmark.py --smoke

# Full 10 sample — 10 baseline + 300 corrupted = 310 cases
python scripts/run_benchmark.py
```

**Da chay thanh cong:** 0 errors, 2440 rows metrics.csv (dung STUB metrics).  
Khi Nguoi 2 implement `src/metrics.py`, chay lai lenh tren de ghi de voi metric that.

### Upstream repo

- URL: https://github.com/thu-ml/3D_Corruptions_AD  
- Local path: `3D_Corruptions_AD/`  
- Commit: `48c23f77fe82beab599f8248b7794928334a3fb5`  
- License: MIT — Copyright (c) 2023 Tsinghua Machine Learning Group  

---

## Nhiem vu Nguoi 2 (con lai)

### 1. Implement `src/metrics.py`

Bo `raise NotImplementedError` va viet:

```python
def evaluate(clean, corrupted, kind, range_bins):
    # clean, corrupted: np.ndarray float32 (N, 4)
    # kind: "baseline" | "density_decrease" | "gaussian_noise"
    # range_bins: list[float], e.g. [0, 10, 20, 40, 80]
    # Tra ve: list[dict] voi keys: metric, range_min_m, range_max_m, value, unit
    ...
```

Contract day du xem trong `src/metrics.py` va `guild_project.md` muc 7.

**Sau khi implement xong**, chay lai:
```bash
python scripts/run_benchmark.py
```
Runner tu dong phat hien va dung `src.metrics.evaluate` thay STUB.

### 2. Viet `scripts/summarize_results.py`

Doc `outputs/metrics.csv`, tao:
- `outputs/summary_per_sample.csv` — mean/std (ddof=1) qua 3 seed, theo sample/corruption/severity/metric/bin
- `outputs/summary_overall.csv` — trung binh khong trong so qua sample means, bao so sample hop le

### 3. Viet `src/visualization.py` va BEV plots

- BEV baseline vs density severity 5 vs gaussian severity 5 cho >= 3 sample
- Plot retention theo severity/range
- Plot RMSE theo sigma
- Luu PNG vao `outputs/figures/`

### 4. Hoan thien README va `docs/report.md`

- Setup + lenh thuc te
- Claim, method, ket qua, failure case, gioi han
- Khong dien so gia; neu chua chay thi ghi chua chay

---

## Cau truc thu muc

```
guild_project.md          # Huong dan du an
README.md                 # File nay
requirements.txt          # numpy>=1.24
configs/benchmark.json    # Config chay
src/
  data_io.py              # DONE (Nguoi 1)
  corruption_adapter.py   # DONE (Nguoi 1)
  metrics.py              # PENDING (Nguoi 2)
  visualization.py        # PENDING (Nguoi 2)
scripts/
  check_dataset.py        # DONE (Nguoi 1)
  run_benchmark.py        # DONE (Nguoi 1)
  summarize_results.py    # PENDING (Nguoi 2)
docs/
  source_notes.md         # DONE (Nguoi 1)
  report.md               # PENDING (Nguoi 2)
data/
  velodyne/               # 10 x .bin (baseline)
  velodyne_reduced/       # 10 x .bin (CHUA DUNG — nguon goc chua xac minh)
  image/                  # 10 x .png
  calib/                  # 10 x .txt
  label/                  # 10 x .txt
3D_Corruptions_AD/        # Upstream repo clone (MIT)
outputs/
  manifest.csv            # DONE
  metrics.csv             # DONE (STUB — cho real metrics)
  run_metadata.json       # DONE
  summary_per_sample.csv  # PENDING
  summary_overall.csv     # PENDING
  figures/                # PENDING
```

---

## Ghi chu ky thuat (Nguoi 1 ban giao)

- **velodyne_reduced:** Khong dung lam corruption. Ratio size ~16% so velodyne — nguon goc chua ro (co the la camera-FOV crop). Xem them `docs/source_notes.md`.
- **Gaussian adaptation:** Upstream them noise vao tat ca C cot. Adapter nay chi them noise vao XYZ, giu nguyen intensity. Ghi ro trong `src/corruption_adapter.py`.
- **Severity 0:** Khong truyen vao ham upstream. Baseline la clean cloud.
- **Seed:** ap dung qua `np.random.seed()` vi upstream dung numpy global state.
- **Khong luu 300 corrupted cloud:** Chi luu so do va metadata. Tai tao bang seed/config.