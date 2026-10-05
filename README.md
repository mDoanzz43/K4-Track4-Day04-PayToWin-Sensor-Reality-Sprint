# K4-Track4-Day04 — Lab T2 LiDAR Corruption Benchmark

**Nhóm:** PayToWin

| STT | Thành viên | MSSV | Phân công |
|---:|---|---|---|
| 1 | Đỗ Mạnh Đoan | 2A202602839 | Data, corruption, runner, config |
| 2 | Nguyễn Mạnh Cường | 2A2026.... | Metrics, summary, visualization, report |

## Kết quả

Pipeline đã hoàn thành và chạy full trên 10 scan: **10 baseline + 300 corrupted cases, 0 lỗi**. `outputs/metrics.csv` có 3.410 dòng metric thật, không còn STUB.

| Severity | Global density retention (%) | Global XYZ RMSE (m) |
|---:|---:|---:|
| 1 | 94.0003 | 0.03465 |
| 2 | 88.0004 | 0.06931 |
| 3 | 82.0005 | 0.10396 |
| 4 | 76.0006 | 0.13862 |
| 5 | 70.0004 | 0.17327 |

Đây là metric dữ liệu cảm biến, chưa phải object recall/mAP hoặc kết luận về an toàn. Phương pháp, failure case và giới hạn được trình bày trong [docs/report.md](docs/report.md).

## Cài đặt

Yêu cầu Python >= 3.9.

```bash
pip install -r requirements.txt
pip install "matplotlib>=3.7"
```

NumPy dùng cho benchmark và summary; Matplotlib chỉ dùng để sinh hình headless. Không cần Open3D, h5py hoặc model checkpoint.

## Chạy tái hiện

Từ thư mục gốc project:

```bash
# Kiểm tra đủ ID và schema KITTI N x 4
python scripts/check_dataset.py

# Unit check metric: boundary, empty bin, known RMSE
python -m src.metrics

# Smoke một sample hoặc full 310 cases
python scripts/run_benchmark.py --smoke
python scripts/run_benchmark.py

# Tổng hợp và sinh hình
python scripts/summarize_results.py
python -m src.visualization
```

Full run phải được chạy sau smoke vì hai lệnh cùng ghi `outputs/metrics.csv` và `outputs/run_metadata.json`.

## Thiết kế benchmark

- Baseline dùng `data/velodyne`; severity 0 không truyền vào upstream.
- Corruption: `density_decrease` và `gaussian_noise`, severity 1–5, seed 42/43/44.
- Density xóa xấp xỉ 6/12/18/24/30% số điểm.
- Gaussian dùng sigma 0.02/0.04/0.06/0.08/0.10 m trên XYZ, giữ intensity và thứ tự.
- Range bin: `[0,10)`, `[10,20)`, `[20,40)`, `[40,80)`, và `[80,+∞)` m.
- Metric: `point_count`, `retention_percent`, `xyz_rmse_m`.
- Bin clean rỗng trả NA; summary không thay NA bằng 0.
- `summary_per_sample.csv`: mean/std mẫu (`ddof=1`) qua seed.
- `summary_overall.csv`: mean không trọng số và std giữa sample, kèm `n_valid_samples`.

## Outputs

| File/thư mục | Nội dung |
|---|---|
| `outputs/manifest.csv` | Kiểm tra 10 ID và file đầu vào |
| `outputs/metrics.csv` | 3.410 dòng metric theo case/bin |
| `outputs/summary_per_sample.csv` | 1.210 dòng tổng hợp qua seed |
| `outputs/summary_overall.csv` | 121 dòng tổng hợp qua sample |
| `outputs/run_metadata.json` | Config, checksum, phiên bản, provenance, lỗi |
| `outputs/figures/` | 3 BEV + retention plot + RMSE plot |

BEV dùng ba sample đầu theo thứ tự ID (`000000`, `000001`, `000002`), severity 5, seed 42 và cùng axes. Sampling hiển thị là deterministic; mọi metric dùng toàn bộ điểm.

## Nguồn và adaptation

- Upstream: https://github.com/thu-ml/3D_Corruptions_AD
- Commit: `48c23f77fe82beab599f8248b7794928334a3fb5`
- License: MIT — Copyright (c) 2023 Tsinghua Machine Learning Group
- `density_dec_global` giữ nguyên logic upstream.
- Gaussian được điều chỉnh chỉ làm nhiễu XYZ vì upstream làm nhiễu toàn bộ cột, gồm intensity khi input là N×4.

`data/velodyne_reduced` không được dùng vì nguồn gốc/cách tạo chưa xác minh. Chi tiết ở [docs/source_notes.md](docs/source_notes.md).

## Cấu trúc chính

```text
configs/benchmark.json
src/data_io.py
src/corruption_adapter.py
src/metrics.py
src/visualization.py
scripts/check_dataset.py
scripts/run_benchmark.py
scripts/summarize_results.py
docs/source_notes.md
docs/report.md
outputs/
```
