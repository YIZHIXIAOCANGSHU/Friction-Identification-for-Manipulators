# Friction Identification Core

当前实现已经切到统一 pipeline 架构，仿真和真机共享同一套采集与辨识核心。

## 目录

- `default.yaml`
  - 统一配置模板
- `config.py`
  - YAML 配置加载
- `trajectory.py`
  - 五次多项式与逐电机激励轨迹
- `controller.py`
  - 前馈 + PD 控制器、安全保护、补偿参数加载
- `estimator.py`
  - 摩擦参数估计
- `results.py`
  - 聚合结果读写
- `pipeline.py`
  - 统一辨识流程
- `sources/`
  - 仿真/真机数据源
- `mujoco_env.py`
  - MuJoCo 环境封装
- `serial_protocol.py`
  - 串口协议与帧收发
- `runtime.py`
  - 输出与结果文件工具
- `visualization.py`
  - Rerun 可视化与真机姿态显示
- `mujoco_support.py`
  - MuJoCo 模型/场景构建辅助

## 运行

```bash
./run.sh
./run.sh sim-ff
./run.sh hw-comp
python3 -m friction_identification_core run --source sim
python3 -m friction_identification_core run --source sim --mode full_feedforward
python3 -m friction_identification_core run --source hw --mode collect
python3 -m friction_identification_core run --source hw --mode compensate
python3 -m friction_identification_core run --source hw --mode full_feedforward
```
