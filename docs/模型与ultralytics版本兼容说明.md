# 模型与 ultralytics 版本兼容说明

## 错误：'Conv' object has no attribute 'bn'

出现该错误时，说明当前安装的 **ultralytics** 与用来训练/保存 `best.pt` 的版本不一致，`Conv` 模块结构不同（新版可能已无 `.bn` 或已融合）。

## 解决思路

### 方案一：降级 ultralytics（优先尝试）

若 `best.pt` 是较早版本 ultralytics 训练得到的，可先尝试安装兼容的旧版本：

```bash
pip uninstall ultralytics -y
pip install ultralytics==8.0.200
```

然后**重启 Web 服务**，再上传 cm1 测试检测。

若 8.0.200 仍报错，可再试：

```bash
pip install ultralytics==8.1.0
```

### 方案二：用当前版本重新训练

若降级后仍有问题，或你希望长期使用最新 ultralytics，建议用**当前环境**重新训练抻面模型，得到与当前库兼容的 `best.pt`：

```bash
# 使用当前已标注数据（含 cm1～cm7 及 cm10～cm12）重新训练
python src/training/train_detection_model.py --videos cm1 cm2 cm3 cm4 cm5 cm6 cm7 cm10 cm11 cm12
```

训练完成后，新生成的 `best.pt` 会与当前 ultralytics 一致，不再出现 `Conv.bn` 类错误。

### 方案三：确认训练时的版本

若你记得当时训练时用的 ultralytics 版本，可直接安装该版本，例如：

```bash
pip install ultralytics==<当时使用的版本号>
```

---

**建议顺序**：先按方案一尝试 `ultralytics==8.0.200`；若检测正常则保持该版本；若仍报错再考虑方案二重新训练。
