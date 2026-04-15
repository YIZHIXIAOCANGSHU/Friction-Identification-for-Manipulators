# OpenArm 摩擦力辨识

这个项目用于在 MuJoCo 仿真中采集 OpenArm 左臂关节数据，并拟合每个关节的摩擦参数。

如果你是第一次接触这个工程，可以直接记住一条命令：

```bash
./run.sh
```

它会自动检查 Python、安装依赖，并启动摩擦力辨识流程。

## 这个项目现在保留了什么

为了方便使用，工程已经精简为只保留摩擦力辨识相关内容：

- `friction_identification_core/`
  - 摩擦数据采集、参数拟合、Rerun 可视化
- `openarm_mujoco/`
  - OpenArm 的 MuJoCo 模型和网格资源
- `results/`
  - 运行后生成的结果文件
- `run.sh`
  - 推荐入口脚本

## 运行前需要准备什么

建议环境：

- Linux
- Python 3.10 或更高版本
- 可以正常导入 `mujoco`

如果你希望看到可视化界面，还需要本机图形环境支持：

- `--render` 会打开 MuJoCo 窗口
- `--spawn-rerun` 会打开 Rerun 窗口

## 第一次使用

### 1. 进入项目目录

```bash
cd /home/luwen/桌面/参数4.8_摩擦力辨识
```

### 2. 安装依赖

如果你想手动安装依赖：

```bash
pip3 install -r requirements.txt
```

依赖很少，主要包括：

- `mujoco`
- `numpy`
- `scipy`
- `rerun-sdk`

### 3. 直接运行

```bash
./run.sh
```

程序会做这些事：

1. 检查 `python3` 是否可用
2. 安装或同步依赖
3. 加载 OpenArm 的 MuJoCo 模型
4. 生成激励轨迹并采集各关节摩擦相关力矩
5. 拟合每个关节的库仑摩擦、粘性摩擦和偏置项
6. 保存结果到 `results/`

## 常用命令

最常用的几种运行方式如下。

### 只运行辨识

```bash
./run.sh
```

### 辨识时显示 MuJoCo 画面

```bash
./run.sh --render
```

### 辨识结束后查看 Rerun 可视化

```bash
./run.sh --spawn-rerun
```

### 同时打开 MuJoCo 和 Rerun

```bash
./run.sh --render --spawn-rerun
```

### 调整采集时长或采样率

```bash
./run.sh --duration 24 --sample-rate 400
```

## 参数说明

`run.sh` 会把参数原样传给摩擦辨识主程序，常用参数有：

- `--duration`
  - 激励与采集总时长，单位秒
- `--sample-rate`
  - 记录频率，单位 Hz
- `--base-frequency`
  - 激励轨迹的基频
- `--amplitude-scale`
  - 轨迹激励幅值缩放
- `--feedback-scale`
  - 轨迹跟踪时的 PD 反馈比例
- `--render`
  - 打开 MuJoCo 可视化
- `--spawn-rerun`
  - 打开 Rerun 可视化

查看帮助：

```bash
./run.sh --help
```

## 输出结果怎么看

运行完成后，结果会保存到 `results/`：

- `results/friction_identification_summary.json`
  - 每个关节的真值与估计值
  - 训练集和验证集误差
  - 总体平均验证指标
- `results/friction_identification_data.npz`
  - 原始时间序列数据
  - 关节速度
  - 摩擦力矩
  - 预测结果
  - 训练/验证样本掩码

终端还会打印每个关节的主要结果，例如：

- 估计的库仑摩擦 `fc`
- 估计的粘性摩擦 `fv`
- 验证集 `RMSE`

## 新手如何判断运行是否成功

一般来说，满足下面几项就说明流程跑通了：

- 终端出现 `Friction identification finished.`
- `results/` 下生成了两个结果文件
- 没有出现 `Failed to load MuJoCo` 或 Python 导入报错

如果你启用了 `--render` 或 `--spawn-rerun`，还能看到对应窗口被正常拉起。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── run.sh
├── friction_identification_core/
│   ├── estimator.py
│   ├── models.py
│   ├── mujoco_driver.py
│   ├── rerun_reporter.py
│   └── run_openarm_friction_identification.py
├── openarm_mujoco/
│   ├── scene_with_target_gripper.xml
│   ├── openarm_bimanual_gripper.xml
│   └── meshes/
└── results/
```

## 如果运行失败，可以先检查这些

### 1. `python3` 不存在

先确认 Python 是否安装：

```bash
python3 --version
```

### 2. `mujoco` 导入失败

尝试重新安装依赖：

```bash
pip3 install -r requirements.txt
```

### 3. 图形窗口打不开

如果你使用了：

- `--render`
- `--spawn-rerun`

那就需要本机有可用显示环境。没有图形环境时，可以先只运行：

```bash
./run.sh
```

### 4. 结果文件没有生成

优先检查终端最后几行输出，看是否在采集、拟合或写文件阶段报错。

## 想继续看代码时，从哪里开始

推荐阅读顺序：

1. `run.sh`
2. `friction_identification_core/run_openarm_friction_identification.py`
3. `friction_identification_core/mujoco_driver.py`
4. `friction_identification_core/estimator.py`

这样最容易看清“入口 -> 数据采集 -> 参数拟合”的主流程。
