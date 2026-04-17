# OpenArm 摩擦力辨识

项目现在按 `plans/simplify_architecture_plan.md` 收敛为统一 pipeline 架构，旧的兼容包装层已经移除，只保留一套顶层实现：

- `friction_identification_core/__main__.py`
  - 统一 CLI 入口
- `friction_identification_core/config.py`
  - YAML 配置加载
- `friction_identification_core/default.yaml`
  - 默认配置
- `friction_identification_core/pipeline.py`
  - 仿真/真机共享的辨识流程
- `friction_identification_core/sources/`
  - `simulation.py` / `hardware.py` 数据源实现
- `friction_identification_core/controller.py`
  - 控制、安全、补偿
- `friction_identification_core/trajectory.py`
  - 轨迹生成
- `friction_identification_core/estimator.py`
  - 摩擦参数估计
- `friction_identification_core/results.py`
  - 单文件结果管理
- `friction_identification_core/mujoco_env.py`
  - MuJoCo 环境封装
- `friction_identification_core/serial_protocol.py`
  - 串口协议
- `friction_identification_core/visualization.py`
  - 可视化
- `friction_identification_core/mujoco_support.py`
  - MuJoCo 模型构建辅助

## 快速开始

```bash
./run.sh
```

默认会进入交互式菜单，通过数字选择运行模式，并显示当前配置文件与目标关节。

快捷模式也保留：

```bash
./run.sh sim
./run.sh sim-ff
./run.sh hw
./run.sh hw-comp
./run.sh hw-ff
```

其中 `real` 仍然兼容旧用法，等价于 `./run.sh hw`。

默认配置文件：

```text
friction_identification_core/default.yaml
```

菜单模式下可直接：

- 输入 `1` 启动仿真采集
- 输入 `4` 启动真机补偿验证
- 输入 `j` 临时切换目标关节
- 输入 `c` 切换配置文件
- 输入 `r` 重复上次运行

底层统一 CLI 入口仍然可直接使用。

仿真入口：

```bash
python3 -m friction_identification_core run --source sim --config friction_identification_core/default.yaml
python3 -m friction_identification_core run --source sim --config friction_identification_core/default.yaml --mode full_feedforward
```

真机入口：

```bash
python3 -m friction_identification_core run --source hw --config friction_identification_core/default.yaml --mode collect
python3 -m friction_identification_core run --source hw --config friction_identification_core/default.yaml --mode compensate
python3 -m friction_identification_core run --source hw --config friction_identification_core/default.yaml --mode full_feedforward
```

## 配置方式

所有关键参数都在 YAML 里维护。最常改的是：

```yaml
identification:
  target_joint: 0
```

一次只辨识一个电机，控制器只对该电机输出非零力矩，其他关节力矩为 0。

## 输出结果

结果默认写入 `results/`，核心文件为：

- `simulation_results.npz`
- `hardware_results.npz`

新的 pipeline 会优先更新这两个聚合结果文件。

## 依赖

```bash
pip3 install -r requirements.txt
```

`run.sh` 现在只做按需依赖检查，不会在每次启动时自动执行 `pip install`。

主要依赖：

- `mujoco`
- `numpy`
- `scipy`
- `rerun-sdk`
- `pyserial`
- `PyYAML`
