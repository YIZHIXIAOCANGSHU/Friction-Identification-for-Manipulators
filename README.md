# OpenArm 摩擦力辨识

项目现在按 `plans/refactor_plan.md` 重构为新的分层结构：

- `friction_identification_core/config/`
  - YAML 配置与加载器
- `friction_identification_core/core/`
  - 轨迹、控制器、安全、拟合核心与数据模型
- `friction_identification_core/simulation/`
  - MuJoCo 环境与仿真 runner
- `friction_identification_core/hardware/`
  - 串口协议与真机 runner
- `friction_identification_core/cli/`
  - `simulate.py` / `deploy.py`
- `friction_identification_core/utils/`
  - 日志、可视化与 MuJoCo 场景辅助

## 快速开始

```bash
./run.sh
```

默认会读取：

```text
friction_identification_core/config/default.yaml
```

仿真入口：

```bash
python3 -m friction_identification_core.cli.simulate --config friction_identification_core/config/default.yaml
```

真机入口：

```bash
python3 -m friction_identification_core.cli.deploy --config friction_identification_core/config/default.yaml --mode collect
python3 -m friction_identification_core.cli.deploy --config friction_identification_core/config/default.yaml --mode compensate
```

## 配置方式

所有关键参数都在 YAML 里维护。最常改的是：

```yaml
identification:
  target_joint: 0
```

一次只辨识一个电机，控制器只对该电机输出非零力矩，其他关节力矩为 0。

## 输出结果

结果默认写入 `results/`，包括：

- `friction_identification_joint_<n>.npz/.json`
- `real_uart_capture_<mode>_joint_<n>.npz/.json`
- `real_friction_identification_joint_<n>.npz/.json`
- `real_friction_identification_summary.json`

## 依赖

```bash
pip3 install -r requirements.txt
```

主要依赖：

- `mujoco`
- `numpy`
- `scipy`
- `rerun-sdk`
- `pyserial`
- `PyYAML`
