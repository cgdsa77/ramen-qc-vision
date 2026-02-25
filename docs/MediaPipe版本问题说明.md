# MediaPipe版本问题说明

## 问题描述

当前安装的MediaPipe版本是0.10.31，该版本的API与旧版本（<0.10）发生了重大变化：
- 旧版本使用 `mediapipe.solutions.pose`
- 新版本使用 `mediapipe.tasks.python` 的tasks API

## 解决方案

### 方案1：使用旧版本MediaPipe（推荐）

如果PyPI上还有旧版本可用，可以安装：

```bash
pip install 'mediapipe<0.10'
```

如果无法安装旧版本，可以考虑：

### 方案2：使用其他姿态估计库

1. **OpenPose**：功能强大但安装复杂
2. **MMPose**：更专业的姿态估计框架
3. **YOLO-Pose**：基于YOLO的姿态估计

### 方案3：等待代码更新

代码需要更新以支持MediaPipe 0.10+的新API。

## 临时解决方案

如果需要立即测试，可以考虑：
1. 使用其他姿态估计库（如MMPose）
2. 或者先使用目标检测的结果进行初步分析

## 当前状态

代码中的`src/models/pose.py`当前只支持MediaPipe旧版API（<0.10）。

如果要使用MediaPipe 0.10+，需要重写该模块以使用新的tasks API。