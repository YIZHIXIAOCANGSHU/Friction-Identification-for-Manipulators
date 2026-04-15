# Friction Identification Core

当前实现已经切到 YAML 配置驱动的新架构。

## 目录

- `config/default.yaml`
  - 统一配置模板
- `config/loader.py`
  - YAML 加载与 dataclass 映射
- `core/trajectory.py`
  - 五次多项式与逐电机激励轨迹
- `core/controller.py`
  - 前馈 + PD 控制器
- `core/safety.py`
  - 关节限位检测与力矩限幅
- `simulation/mujoco_env.py`
  - MuJoCo 环境封装
- `simulation/runner.py`
  - 仿真辨识流程
- `hardware/runner.py`
  - 真机串口采集/补偿流程
- `hardware/serial_protocol.py`
  - 串口协议与帧收发
- `cli/simulate.py`
  - 仿真入口
- `cli/deploy.py`
  - 真机入口
- `utils/logging.py`
  - 输出与结果文件工具
- `utils/visualization.py`
  - Rerun 可视化与真机姿态显示
- `utils/mujoco.py`
  - MuJoCo 模型/场景构建辅助

## 运行

```bash
python3 -m friction_identification_core.cli.simulate
python3 -m friction_identification_core.cli.deploy --mode collect
```
