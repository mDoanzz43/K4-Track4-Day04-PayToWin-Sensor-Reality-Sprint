# docs/source_notes.md -- Nguon du lieu va ghi chu ky thuat

## Dataset

- **Nguon**: KITTI tracking/object dataset (xac nhan qua dinh dang file calib: P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo)
- **Schema xac minh (000000.bin)**:
  - Dinh dang: float32 little-endian, N x 4 (x, y, z, intensity)
  - File size % 16 == 0 (dieu kien can, da kiem tra toan bo 10 file)
  - x: [-71.0, +73.0] m, y: [-21.1, +53.8] m, z: [-5.2, +2.7] m
  - intensity: [0.0, 1.0] (normalized)
  - Tat ca gia tri finite: True
  - Don vi XYZ: met (phu hop KITTI spec)

## Upstream Repo

- **URL**: https://github.com/thu-ml/3D_Corruptions_AD
- **Clone local**: 3D_Corruptions_AD/
- **Remote**: origin https://github.com/thu-ml/3D_Corruptions_AD.git
- **HEAD commit**: 48c23f77fe82beab599f8248b7794928334a3fb5
- **License**: MIT -- Copyright (c) 2023 Tsinghua Machine Learning Group
- **Paper**: Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving, CVPR 2023

## Hai ham su dung tu upstream

### density_dec_global (verbatim)
- Xoa ngau nhien c diem tu N diem
- num = int(N * 0.3)
- c = [int(0.2*num), int(0.4*num), int(0.6*num), int(0.8*num), num][severity-1]
- Ty le xoa thuc te: ~6%, ~12%, ~18%, ~24%, ~30% (co lam tron int)
- Dung np.random.choice + np.delete; su dung numpy random global state
- Khong mutate theo nghia lam moi array (np.delete tra ve array moi)

### gaussian_noise (co ADAPTATION)
- UPSTREAM: them noise N(0, sigma) vao TAT CA C cot (ke ca intensity khi C=4)
- ADAPTATION cua nhom: chi them noise vao [:, :3] (XYZ); giu nguyen cot intensity
- Sigma: [0.02, 0.04, 0.06, 0.08, 0.10] m tuong ung severity 1-5
- Giu nguyen so diem va thu tu (shape va index khong doi)
- Ghi chu trong code: "upstream comment says N*3 suggesting original designed for 3-col"

## velodyne_reduced -- CHUA XAC MINH

- **Kich thuoc so sanh** (size_bytes reduced / size_bytes velodyne):
  - 000000: 324560 / 1846144 = 0.176
  - 000001: 298080 / 1924288 = 0.155
  - 000002: 323360 / 2030256 = 0.159
  - Trung binh: ~16-18% so voi velodyne day du
- **Gia thiet**: co the la cloud gioi han trong camera FOV (thong thao trong KITTI pipeline)
- **KHONG SU DUNG**: khong dung velodyne_reduced lam corruption hoac so sanh severity
- **Can lam them**: tim script hoac thu muc nguon de xac nhan cach tao reduced

## Dependency

- Python >= 3.9 (test voi 3.10.0)
- numpy >= 1.24 (test voi 2.2.5)
- KHONG can: open3d, h5py, distortion (chi la transitive import trong LiDAR_corruptions.py goc)

## Ghi chu ky thuat khac

- Seed: ap dung qua np.random.seed() vi upstream dung np.random toan cuc.
  apply_corruption tao RandomState(seed) -> lay seed con -> dat np.random.seed.
- Severity 0 KHONG truyen vao upstream; baseline la clean cloud.
- CSV schema: sample_id, corruption, severity, seed, metric, range_min_m, range_max_m, value, unit
- Corruption name thong nhat: baseline, density_decrease, gaussian_noise
- Khong luu 300 corrupted cloud; chi luu so do va metadata.