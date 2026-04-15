# Friction Identification Core

这个工程已经收口为只保留 OpenArm 摩擦力辨识相关代码。

## 直接运行

```bash
./run.sh
```

常用参数：

```bash
./run.sh --render
./run.sh --spawn-rerun
./run.sh --duration 24 --sample-rate 400
```

结果默认输出到：

```text
results/
```

## 目录说明

- `estimator.py`
  - 多关节摩擦参数拟合
  - 默认模型：`tau_f = fc * tanh(qd / vs) + fv * qd + bias`
- `mujoco_driver.py`
  - 激励轨迹生成、MuJoCo 仿真与摩擦力矩采集
- `rerun_reporter.py`
  - Rerun 可视化
- `run_openarm_friction_identification.py`
  - OpenArm 端到端运行入口

## 可复用接口

```python
from friction_identification_core.estimator import (
    fit_multijoint_friction,
    predict_friction_torque,
)
```
