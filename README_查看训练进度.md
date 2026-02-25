# 如何查看训练进度

## 方法1：使用批处理文件（最简单）

直接双击 `查看训练进度.bat` 文件即可。

## 方法2：使用命令行

在项目根目录打开命令行（PowerShell 或 CMD），运行：

```bash
python scripts/check_training_progress.py
```

## 方法3：在 Cursor 终端中运行

在 Cursor 的终端中，确保当前目录是项目根目录，然后运行：

```bash
python scripts/check_training_progress.py
```

## 脚本会显示什么信息？

- **训练进度**：当前已完成多少轮（如 12/150）
- **累计训练时间**：已经训练了多长时间
- **最新指标**：
  - 精确率 (Precision)
  - 召回率 (Recall)
  - mAP50（平均精度，IoU=0.5）
  - mAP50-95（平均精度，IoU=0.5-0.95）
- **训练损失和验证损失**
- **趋势分析**：指标是提升还是下降
- **预计剩余时间**：还需要训练多长时间

## 提示

- 可以随时运行这个脚本查看进度，不会影响训练
- 训练在后台自动进行，不需要保持脚本运行
- 训练完成后，最佳模型会保存在 `models/stretch_detection/weights/best.pt`

