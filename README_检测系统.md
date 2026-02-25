# 抻面动作检测系统使用说明

## 系统概述

这是一个基于Web的视频检测系统，可以：
- 上传视频文件
- 实时检测视频中的目标（手部、面条等）
- 在视频上显示检测框
- 查看检测统计信息

## 快速开始

### 1. 训练检测模型（首次使用）

基于已标注的3个标准视频（cm1, cm2, cm3）训练检测模型：

```bash
python src/training/train_detection_model.py
```

**说明：**
- 这会基于你的标注数据训练一个YOLO检测模型
- 模型会保存在 `models/stretch_detection_model.pt`
- 训练时间取决于数据集大小（通常需要几分钟到几十分钟）

**注意事项：**
- 如果训练数据较少（只有3个视频），模型可能不够准确
- 建议先测试效果，如果检测不准确，继续标注更多视频后重新训练

### 2. 启动Web服务器

```bash
python scripts/start_server.py
```

或者：

```bash
python src/main.py --mode api --port 8000
```

### 3. 打开Web界面

在浏览器中访问：
```
http://localhost:8000
```

### 4. 使用界面

1. **上传视频**
   - 点击"选择视频文件"按钮
   - 或直接拖拽视频文件到上传区域
   - 支持格式：MP4, AVI, MOV等

2. **查看检测结果**
   - 上传后系统会自动处理视频
   - 处理完成后会显示原始视频和检测结果
   - 点击"开始检测"播放视频，可以看到实时检测框

3. **查看统计信息**
   - 界面下方会显示检测统计
   - 包括总帧数、各类别检测次数等

## 检测类别

系统会检测以下类别（基于你的标注）：
- **hand** (手部) - 红色框
- **noodle_rope** (面条) - 蓝色框  
- **noodle_bundle** (面条束) - 绿色框
- **pot_or_table** (锅/桌子) - 黄色框

## 文件结构

```
ramen-qc-vision/
├── web/
│   └── video_detection.html      # Web界面
├── src/
│   ├── api/
│   │   └── video_detection_api.py  # 检测API
│   ├── training/
│   │   └── train_detection_model.py # 训练脚本
│   └── pipelines/
│       └── api_server.py          # API服务器
├── models/
│   └── stretch_detection_model.pt # 训练好的模型
└── scripts/
    └── start_server.py            # 启动脚本
```

## 迭代改进流程

### 当前状态
- ✅ 基准模型：基于 cm1, cm2, cm3（3个标准视频）
- ⏳ 待测试：cm4（上传测试）

### 改进步骤

1. **测试当前模型**
   - 上传 cm4 视频
   - 查看检测效果
   - 评估准确性

2. **如果检测不准确**
   - 继续标注更多视频（cm4, cm5等）
   - 重新训练模型
   - 重复测试直到满意

3. **模型优化**
   - 增加训练轮数（修改 `train_detection_model.py` 中的 `epochs` 参数）
   - 调整置信度阈值
   - 使用更大的模型（yolov8s, yolov8m等）

## 常见问题

### Q: 模型检测不准确怎么办？
A: 
1. 检查是否有足够的训练数据（至少需要标注5-10个视频）
2. 增加训练轮数（epochs）
3. 确保标注质量
4. 考虑使用更大的模型

### Q: 如何重新训练模型？
A: 
```bash
python src/training/train_detection_model.py
```
训练完成后重启服务器即可使用新模型。

### Q: 可以使用预训练模型吗？
A: 系统会自动使用YOLOv8预训练模型，但它只能检测通用类别（如person），无法检测你自定义的类别（hand, noodle_rope等）。必须基于标注数据训练自定义模型。

## 下一步

1. **当前阶段**：测试检测效果
   - 上传 cm4 视频查看检测结果
   - 评估是否需要更多训练数据

2. **中期目标**：优化检测模型
   - 标注更多视频（cm4, cm5, cm6等）
   - 不断训练和改进模型
   - 提高检测准确率

3. **长期目标**：完整的分析系统
   - 检测模型稳定后，构建评分模型
   - 添加"下面"和"捞面"的检测
   - 生成综合分析报告

## 技术说明

- **检测框架**: YOLOv8 (ultralytics)
- **Web框架**: FastAPI
- **前端**: HTML5 + JavaScript
- **视频处理**: OpenCV

