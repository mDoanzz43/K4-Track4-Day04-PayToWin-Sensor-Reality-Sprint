# Báo cáo — LiDAR Corruption Benchmark

## 1. Claim trong phạm vi phép đo

Trên 10 scan LiDAR được cung cấp, phép giảm mật độ làm số điểm còn lại giảm theo đúng tỷ lệ đã cấu hình, còn nhiễu Gaussian làm sai lệch vị trí XYZ tăng gần tuyến tính theo sigma. Đây là kết quả ở mức **dữ liệu cảm biến**; benchmark này chưa chứng minh mức giảm object recall, mAP, chất lượng SLAM hay độ an toàn của xe.

## 2. Phương pháp

- Dữ liệu: 10 file `data/velodyne/*.bin`, mỗi điểm là float32 little-endian `(x, y, z, intensity)`.
- Ma trận: 10 baseline và `10 sample × 2 corruption × 5 severity × 3 seed = 300` trường hợp corrupted.
- Seed: 42, 43, 44. Severity 0 chỉ là baseline và không truyền vào hàm upstream.
- Density decrease: xóa xấp xỉ 6%, 12%, 18%, 24%, 30% số điểm ở severity 1–5.
- Gaussian: sigma tọa độ 0.02, 0.04, 0.06, 0.08, 0.10 m; chỉ thay XYZ, giữ intensity, số điểm và thứ tự.
- Range: khoảng cách Euclidean từ gốc LiDAR. Bin `[0,10)`, `[10,20)`, `[20,40)`, `[40,80)` m và bin ngoài miền `[80,+∞)`.
- Metric: point count, density retention và row-wise XYZ RMSE. RMSE không được ghép hàng sau dropout.
- Tổng hợp: mean/std mẫu (`ddof=1`) qua ba seed cho từng sample; sau đó mean không trọng số và std giữa 10 sample. NA bị loại khỏi thống kê, không thay bằng 0.

Nguồn corruption là `3D_Corruptions_AD`, commit `48c23f77fe82beab599f8248b7794928334a3fb5`, giấy phép MIT. Gaussian được điều chỉnh so với upstream để không làm nhiễu intensity.

## 3. Kết quả thực đo

Full run ngày 2026-10-05 hoàn thành 10 baseline + 300 corrupted cases, 0 lỗi, tạo 3.410 dòng trong `metrics.csv`. Không còn unit STUB.

| Severity | Điểm giữ lại toàn cục (%) | XYZ RMSE toàn cục (m) |
|---:|---:|---:|
| 1 | 94.0003 | 0.03465 |
| 2 | 88.0004 | 0.06931 |
| 3 | 82.0005 | 0.10396 |
| 4 | 76.0006 | 0.13862 |
| 5 | 70.0004 | 0.17327 |

Các giá trị là mean không trọng số qua 10 sample means. Ở severity 5, retention trung bình theo range lần lượt là 69.9324%, 70.0352%, 70.0422% và 70.2835% cho bốn bin 0–80 m. Sai khác nhỏ giữa bin phù hợp với việc xóa điểm ngẫu nhiên toàn cục. RMSE severity 5 theo bốn bin là 0.17317, 0.17336, 0.17338 và 0.17357 m; global 0.17327 m, sát kỳ vọng lý thuyết `sqrt(3) × 0.10 = 0.17321 m`.

Tất cả 10 scan không có điểm ở range `>=80 m`; vì vậy 310 giá trị metric ratio/RMSE cho bin này là NA. Đây là xử lý đúng contract, không phải lỗi chạy.

## 4. Bằng chứng trực quan

Ba sample `000000`, `000001`, `000002` được chọn theo thứ tự ID, không chọn theo hình đẹp. BEV so sánh baseline, density severity 5 và Gaussian severity 5 tại seed 42, dùng cùng axes/limits/aspect. Nếu cloud vượt 50.000 điểm, hình dùng stride cố định chỉ để render; metric luôn dùng toàn bộ cloud.

- `outputs/figures/bev_000000.png`
- `outputs/figures/bev_000001.png`
- `outputs/figures/bev_000002.png`
- `outputs/figures/retention_by_severity_range.png`
- `outputs/figures/rmse_by_sigma.png`

Error bar trong hai plot tổng hợp là std **giữa scene**, không phải độ bất định theo seed.

## 5. Failure case quan sát được

Ở sample `000000`, density severity 5 trong bin 40–80 m giữ lại 72.0976%, 73.8537%, 70.7317% theo seed 42/43/44; mean 72.2276%, std theo seed 1.5650%. Đây là std theo seed lớn nhất trong 10 sample ở bin xa tại severity 5. Hình `bev_000000.png` cho thấy cloud bị thưa đi, còn số đo cho thấy cùng tỷ lệ xóa toàn cục vẫn tạo dao động cục bộ đáng kể hơn ở vùng ít điểm.

Failure này chưa cho biết vật thể nào bị bỏ sót. Một fallback có thể kiểm chứng là cảnh báo khi retention trong một bin quan trọng xuống dưới ngưỡng hiệu chỉnh trên validation set, sau đó giảm tốc hoặc chuyển sang cảm biến khác. Cần đo thêm false-alarm rate của cảnh báo và detector recall/mAP theo range trước khi khẳng định hiệu quả an toàn.

## 6. Giới hạn và hướng tiếp theo

- Chỉ có 10 scan và corruption tổng hợp; chưa đại diện đầy đủ mưa, sương, motion distortion hay lỗi phần cứng.
- Point retention và XYZ RMSE là proxy mức sensor, không phải metric perception.
- `velodyne_reduced` bị loại vì nguồn gốc/cách tạo chưa xác minh.
- Label và calibration có sẵn nhưng chưa dùng vì chưa kiểm chứng đầy đủ pipeline hệ tọa độ/box.
- Global retention gần như được quyết định trực tiếp bởi hàm xóa; đóng góp của benchmark là kiểm tra pipeline, phân bố theo range và khả năng tái hiện, không phải phát hiện một quy luật mới.

Bước tiếp theo nên chạy một detector pretrained trên clean/corrupted cloud, báo mAP/recall theo range và ghép các failure với object count/box đã kiểm chứng. Đồng thời thử health score theo density/RMSE và đánh giá ROC hoặc precision–recall của cơ chế fallback.

## 7. Tái hiện

```bash
pip install -r requirements.txt
pip install "matplotlib>=3.7"
python scripts/check_dataset.py
python -m src.metrics
python scripts/run_benchmark.py
python scripts/summarize_results.py
python -m src.visualization
```

Thông tin command, timestamp, checksum input, phiên bản Python/NumPy và lỗi chạy nằm trong `outputs/run_metadata.json`.
