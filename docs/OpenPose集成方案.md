# OpenPose集成方案

## 概述

OpenPose是CMU开发的人体姿态估计库，精度高但安装复杂。本文档提供完整的集成方案。

## 安装方式

### 方案1：使用预编译版本（推荐Windows）

1. **下载OpenPose**
   - 访问：https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
   - 下载Windows预编译版本（约500MB）

2. **解压到项目目录**
   ```
   ramen-qc-vision/
   └── weights/
       └── openpose/  # 解压到这里
   ```

3. **安装Python包**
   ```bash
   pip install pyopenpose
   ```

4. **设置环境变量**
   ```bash
   # Windows PowerShell
   $env:OPENPOSE_DIR = "C:\path\to\openpose"
   
   # 或添加到系统环境变量
   ```

### 方案2：使用Docker（推荐Linux/Mac）

```bash
docker pull cmuopenpose/openpose
docker run -v $(pwd):/data cmuopenpose/openpose \
  --video /data/video.mp4 --write_json /data/output/
```

### 方案3：从源码编译（最复杂）

参考官方文档：https://github.com/CMU-Perceptual-Computing-Lab/openpose

## 使用方式

### 1. 提取手部关键点

```bash
python scripts/extract_hand_keypoints_openpose.py
```

这会处理cm1-cm3视频，生成JSON格式的关键点数据。

### 2. 在Web界面中使用

修改`start_web.py`，添加OpenPose支持：

```python
@app.post("/api/process_frame_openpose")
async def process_frame_openpose(video_name: str, frame_index: int, file: UploadFile):
    # 使用OpenPose处理
    ...
```

### 3. 可视化

```bash
python scripts/visualize_hand_pose_openpose.py
```

## OpenPose vs MediaPipe

| 特性 | OpenPose | MediaPipe |
|------|----------|-----------|
| 精度 | ⭐⭐⭐⭐⭐ 很高 | ⭐⭐⭐⭐ 较高 |
| 速度 | ⭐⭐ 较慢（5-15 FPS） | ⭐⭐⭐⭐⭐ 很快（30-60 FPS） |
| 安装 | ⭐⭐ 复杂 | ⭐⭐⭐⭐⭐ 简单 |
| 手部关键点 | 21个 | 21个 |
| 多人检测 | ✅ 支持 | ❌ 不支持 |
| 实时性 | ❌ 不适合 | ✅ 适合 |

## 推荐方案

### 如果必须使用OpenPose：

1. **离线预处理**：使用OpenPose批量处理视频，生成JSON
2. **前端读取**：前端直接读取JSON，绘制骨架线
3. **优点**：精度最高，无实时延迟
4. **缺点**：需要预处理，文件较大

### 如果追求实时性：

1. **使用MediaPipe**（已安装）
2. **或使用MMPose**：`pip install mmpose`
3. **优点**：速度快，实时性好
4. **缺点**：精度略低于OpenPose

## 快速开始

### 1. 检查OpenPose是否安装

```bash
python scripts/install_openpose.py
```

### 2. 如果未安装，选择替代方案

**推荐：使用MMPose（精度高，安装简单）**

```bash
pip install mmpose mmcv
```

然后修改代码使用MMPose。

### 3. 如果必须使用OpenPose

按照"方案1：使用预编译版本"进行安装。

## 常见问题

### Q1: OpenPose安装失败怎么办？

A: 使用替代方案：
- MediaPipe（已安装）
- MMPose（推荐）

### Q2: OpenPose速度太慢？

A: 
1. 使用GPU版本（需要CUDA）
2. 降低分辨率
3. 使用离线预处理

### Q3: 如何提高检测精度？

A:
1. 使用OpenPose（精度最高）
2. 或使用MMPose + 高精度模型
3. 增加ROI检测（基于hand标签）

## 下一步

1. 如果选择OpenPose：按照安装步骤进行
2. 如果选择替代方案：使用MMPose或MediaPipe
3. 测试效果后决定最终方案
