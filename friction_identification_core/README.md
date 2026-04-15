# Friction Identification Core

这个工程已经收口为只保留 OpenArm 摩擦力辨识相关代码。

## 直接运行

```bash
./run.sh
```

默认采集时长为 `30s`，终端会显示采集和拟合进度。

核心参数统一放在：

```text
friction_identification_core/config.py
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
  - 带关节限位安全边界的逐关节分段激励轨迹生成、MuJoCo 仿真与摩擦力矩采集
  - 仿真与真机共用的轨迹跟踪控制核心也在这里统一收口
- `config.py`
  - 机器人模型参数、采集参数、筛样参数、拟合参数，以及真机 UART 发送策略的统一配置入口
- `runtime.py`
  - 公共运行时工具，如项目根目录、结果目录、日志与 JSON 写出
- `tracking.py`
  - 仿真辨识后的轨迹跟踪评估与结果序列化
- `real_support.py`
  - 真机 UART 模式下对统一控制核心的复用、起步过渡、安全与结果保存逻辑
- `cli/`
  - 整理后的命令行入口实现
  - `simulation_cli.py` 对应仿真辨识
  - `real_uart_cli.py` 对应真机串口采集/补偿
- `rerun_reporter.py`
  - Rerun 可视化
- `run_simulation.py`
  - 当前主用的仿真入口
- `run_real_uart.py`
  - 当前主用的真机入口
- `run_openarm_friction_identification.py`
  - 兼容旧调用方式的仿真入口包装层
- `run_real_uart_friction.py`
  - 兼容旧调用方式的真机入口包装层

## 可复用接口

```python
from friction_identification_core.estimator import (
    fit_multijoint_friction,
    predict_friction_torque,
)
```
