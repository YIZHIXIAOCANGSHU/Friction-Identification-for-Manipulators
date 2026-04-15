# OpenArm 摩擦力辨识

这个项目只保留 OpenArm 左臂的摩擦力辨识流程。

## 运行

安装依赖：

```bash
pip3 install -r requirements.txt
```

执行辨识：

```bash
./run.sh
```

常用参数：

```bash
./run.sh --render
./run.sh --spawn-rerun
./run.sh --duration 24 --sample-rate 400
```

## 保留内容

- `friction_identification_core/`: 摩擦建模、拟合、采集与可视化
- `openarm_mujoco/`: MuJoCo 模型与网格资源
- `results/`: 输出目录

## 输出

运行完成后会在 `results/` 下生成：

- `friction_identification_summary.json`
- `friction_identification_data.npz`
