# 面向未见物体位姿估计的置信度感知 RGB-D 对应

[![English](https://img.shields.io/badge/README-English-2ea44f?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/README-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-0969da?style=for-the-badge)](README.zh-CN.md)

这是一个小型研究项目，用于在两次 RGB-D 观测之间匹配未见过的物体，将二维匹配提升到三维，并估计相对刚体变换。项目受到 COG 中置信度感知对应思想的启发，但**不是 COG 的精确复现**。

当前实现保持克制，包含：

- OpenCV SIFT/ORB 特征匹配；
- 可选的 DINOv2 特征匹配；
- 明确的 RGB-D 有效性检查和二维到三维反投影；
- 透明、启发式的对应置信度；
- 置信度感知的 Kabsch 优化和 RANSAC；
- 用于检查 D435i 录制序列的只读工具。

对一对观测，主要输出是相对 `4 x 4` 变换，以及匹配数量、RANSAC 内点和残差诊断。静态相机录制得到的变换只能作为对应关系 sanity check，不能自动视为位姿精度测量。

## 当前状态

| 模块 | 状态 | 已实际检查的内容 |
|---|---|---|
| 自定义帧目录加载器 | 已实现 | 单元测试 |
| D435i 序列适配器 | 已实现 | 单元测试和真实序列只读解码 |
| 内参、深度缩放、无效深度、反投影 | 已实现 | 单元测试和真实帧诊断 |
| SIFT 基线 | 已实现 | 真实 D435i 帧上的 CPU 运行 |
| 置信度感知 Kabsch/RANSAC | 已实现 | 确定性的合成数据验证 |
| 录制、运动和桌面深度检查 | 已实现 | 单元测试和只读扫描 |
| 多边形和桌面前景 mask | 已实现（启发式） | 单元测试和真实帧检查；没有分割真值 |
| DINOv2 后端 | 实验性 | 真实数据对的 smoke run；没有真值验证 |
| 加权与非加权比较 | 仅诊断代码 | 尚无有科学意义的结论 |
| 位姿真值 benchmark | 不可用 | 手动旋转没有独立测量 |

### 已验证的合成数据结果

```text
inliers=105/160
weighted_rmse_m=0.003205
rotation_error_deg=0.0839
translation_error_m=0.001614
```

这些数字来自具有已知运动和注入离群点的生成场景，不是真实世界测量结果。

### 已验证的 D435i smoke check

最新的完整录制 `20260820_142523_animebox` 通过了采集程序的完整性检查：

- 600 帧彩色图像和 600 帧深度图像；
- 2,007 个加速度样本和 3,997 个陀螺仪样本；
- RGB/深度帧率约 30 FPS；
- 最大 RGB-深度时间差：10.08 ms；
- 视频队列丢帧为 0，写入错误为 0。

只读适配器、桌面 mask、SIFT 和 DINOv2 路径也已经在选定帧上运行。该录制使用固定相机和手动移动物体，因此变换结果只作为诊断；它不是视角变化 benchmark、真值评估或鲁棒性结论。

## 环境

目标环境为：

- WSL2 Ubuntu 22.04，用于开发和推理；
- 项目内 Python 3.10 环境 `.venv`；
- Windows 端采集对齐的彩色/深度图像；
- 从挂载路径（例如 `/mnt/e/...`）读取已经完成的序列。

CPU 适配器和 SIFT 基线不需要 WSL 直接访问 D435i USB。PyTorch 和 DINOv2 是可选项；不需要 Open3D、ROS、Gazebo 或 Isaac Sim。

Windows 采集程序和原始 RGB-D 数据不属于这里的公开研究工作流。适配器读取已完成的序列时不会写回原始目录。

## 安装

请使用项目内环境，不要安装到系统 Python 或 Conda base 环境。

```bash
source .venv/bin/activate
pip install -e '.[vision,dev]'
```

基本检查：

```bash
python -m pytest -q
python scripts/synthetic_demo.py
python -m pip check
```

## 输入数据

原有的帧目录接口仍然支持：

```text
frame_directory/
├── rgb.png
├── depth_m.npy
└── intrinsics.txt
```

D435i 录制序列包含以下相关文件：

```text
session/
├── calibration.json
├── color/*.png
├── depth/*.png
├── frames.csv
├── imu.csv
├── metadata.json
└── validation_report.json
```

`frames.csv` 是 RGB/深度配对的权威表。适配器从 `calibration.json` 解析对齐后的彩色相机内参，使用记录的深度比例解码原始 `uint16` 深度 PNG，并将原始值 `0` 和 `65535` 映射为无效的 metric depth `0.0`。适配器不会暗中加入 3 m 截断。

## 快速使用

### 检查一帧 D435i 数据

```bash
python scripts/inspect_realsense_sequence.py \
  /mnt/e/path/to/session \
  --frame-index 30 \
  --output results/frame_030
```

该命令会在原始序列目录之外写出 RGB 预览、归一化深度预览和 JSON 摘要。

### 检查一段录制

常规扫描检查文件数量、时间戳、标定、图像解码、深度有效性和采样帧的 SIFT 特征数量：

```bash
python scripts/validate_experiment_candidate.py \
  /mnt/e/path/to/session \
  --full-scan \
  --output results/session_quality.json
```

桌面场景还可以选择性地进行深度平面前景检查。平面点依赖具体场景，必须配合预览人工检查：

```bash
python scripts/validate_experiment_candidate.py \
  /mnt/e/path/to/session \
  --full-scan \
  --tabletop-mask \
  --tabletop-plane-points 80 380 550 380 80 450 550 450 \
  --tabletop-mask-roi 180 100 450 370 \
  --output results/session_quality.json
```

扫描对原始序列是只读的。退出码为 `0=PASS`、`1=WARN`、`2=REJECT`。录制完整性通过，并不等于实验位姿资格已经建立。

### 使用 SIFT 估计一对帧

兼容原有帧目录的调用方式：

```bash
python scripts/run_pair.py \
  data/object/view_000 data/object/view_001 \
  --backend sift \
  --output results/pair.json
```

D435i 序列模式从同一录制中选择两个帧索引：

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend sift \
  --output results/pair.json \
  --evidence-dir results/pair_evidence
```

证据目录包含 RGB/深度预览和匹配图。可以使用矩形 ROI、多边形或预先生成的 mask 限制特征提取；这些操作不会修改 RGB-D 序列。

多边形以扁平的 `x y` 坐标序列传入：

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend sift \
  --reference-polygon 258 129 369 119 391 298 273 327 258 303 \
  --query-polygon 300 117 408 150 415 323 331 341 210 249 254 166 \
  --output results/pair_polygon.json
```

桌面 mask 工具从手动选择的桌面像素拟合平面，并将派生 mask 写到原始序列之外：

```bash
python scripts/generate_tabletop_mask.py \
  /mnt/e/path/to/session \
  --frame-index 80 \
  --plane-points 80 380 550 380 80 450 550 450 \
  --object-roi 180 100 450 370 \
  --output results/tabletop_mask/frame_080
```

将生成的 `mask.png` 通过 `--reference-mask` 和 `--query-mask` 传给配对程序。它是桌面场景前景启发式，不是通用物体分割。

### 可选的 DINOv2 后端

DINOv2 使用相同的图像 ROI、多边形和 mask 输入。目前它是实验性功能，应当作为特征匹配诊断工具使用：

```bash
python scripts/run_pair.py \
  --session /mnt/e/path/to/session \
  --reference-index 80 \
  --query-index 500 \
  --backend dino \
  --dino-model dinov2_vits14 \
  --dino-device cuda \
  --output results/dino_pair.json
```

CUDA 是显式指定的；该命令不会静默回退到 CPU。第一次运行可能会通过 `torch.hub` 下载官方 DINOv2 仓库和权重。没有独立参考运动时，不要把 DINOv2 变换解释为位姿精度。

## 工具说明

| 工具 | 用途 |
|---|---|
| `inspect_realsense_sequence.py` | 解码单帧并保存预览 |
| `validate_experiment_candidate.py` | 检查录制或帧对质量 |
| `export_motion_review.py` | 导出采样 RGB/深度复核图 |
| `approve_motion_review.py` | 记录人工复核过的帧对 |
| `generate_tabletop_mask.py` | 生成场景特定的桌面平面 mask |
| `run_pair.py` | 运行 SIFT 或 DINOv2 对应和位姿估计 |
| `summarize_dino_quality.py` | 汇总已有的 DINOv2 结果 |

运动分数和 mask 报告只是数据选择辅助工具。它们不能判断变化来自手、物体、光照还是相机，也不会生成位姿真值标签。

## 局限性

- 置信度是透明的启发式分数，不是经过校准的概率。
- 当前支持手动 ROI 和多边形；桌面 mask 只适用于特定桌面场景，且是启发式的。
- 低纹理、遮挡、物体对称性和缺失深度仍可能产生看似合理但错误的对应关系。
- 固定相机录制适合解析和 smoke check，不能单独构成视角变化 benchmark。
- 当前没有独立的物体位姿参考。
- 加权与非加权路径目前只是诊断代码，项目不声称其中任一种已有实测优势。
- DINOv2 仍是实验性功能。DINOv3、大规模 benchmark、Open3D、ROS、Gazebo 和 Isaac Sim 暂时明确推迟。

## Future plan

当前的手动旋转采集阶段已经暂停。已有录制可用于验证解析、同步、深度、mask 和特征匹配，但由于名义旋转没有独立测量，不能建立 pose accuracy 结论。

等有更好的调试环境后，下一次受控实验应当：

1. 使用刚性夹具或转台，使物体运动可以重复；
2. 记录目标角度，并尽可能获得独立的参考位姿；
3. 将 ArUco/AprilTag 或夹具参考放在独立的评估路径中，而不是作为匹配特征；
4. 将估计的相对 `SE(3)` 与独立参考进行比较；
5. 之后再讨论 pose accuracy 或加权/非加权性能。

单个平面 marker 在大角度旋转时可能离开视野，因此参考物应固定在夹具上，或使用在整个运动范围内仍有足够可见几何结构的布置。
