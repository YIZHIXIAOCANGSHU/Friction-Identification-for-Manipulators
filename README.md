# OpenArm 摩擦力辨识

这个项目用于在 MuJoCo 仿真中采集 OpenArm 左臂关节数据，并拟合每个关节的摩擦参数。

如果你是第一次接触这个工程，可以直接记住一条命令：

```bash
./run.sh
```

它会自动检查 Python、安装依赖，并启动仿真摩擦辨识流程。
默认会同时打开 MuJoCo 可视化和 Rerun 窗口。
默认采集时长为 `30s`，运行过程中会输出采集与拟合进度。

如果要切到真机串口模式：

```bash
./run.sh real
```

这个模式现在默认是“真机采集辨识模式”：

- 复用 MuJoCo 仿真中的参考轨迹与控制律，不做摩擦补偿
- 仿真和真机现在统一走同一套控制核心：逆动力学前馈 + PD 反馈 + 力矩裁剪 + 限位整形
- 实时接收 UART 反馈
- 用接收到的 7 轴电机状态驱动 MuJoCo 仿真机械臂
- 保存真实采集数据
- 结束后基于真实数据自动做一次摩擦辨识

如果任一接收关节位置超出 `friction_identification_core/config.py` 里的关节限位，程序会立即发送零力矩并触发安全停机。

如果你只想接收状态和记录数据、不想真的向下位机发力矩，可以把 `friction_identification_core/config.py` 里的 `RealUartConfig.send_enabled` 改成 `False`。

如果你想在 `collect` 模式下不发送完整激励、只向真机发送重力+科氏补偿，可以把 `RealUartConfig.send_bias_compensation_only` 改成 `True`。这个开关优先级低于 `send_enabled`：

- `send_enabled=True`
  - 发送完整激励
- `send_enabled=False` 且 `send_bias_compensation_only=True`
  - 只发送重力+科氏补偿
- 两者都关
  - 只接收、计算、记录，不发送

核心参数现在统一收口在 `friction_identification_core/config.py`。

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

- 默认会打开 MuJoCo 窗口
- 默认会打开 Rerun 窗口
- 如果不想打开，可以传 `--no-render` 或 `--no-spawn-rerun`

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
4. 默认拉起 MuJoCo 与 Rerun 可视化窗口
5. 生成带关节限位保护的逐关节分段激励轨迹，并在仿真控制环中额外执行力矩裁剪、限位附近力矩衰减与越界停机，再采集各关节摩擦相关力矩
6. 拟合每个关节的库仑摩擦、粘性摩擦和偏置项
7. 保存结果到 `results/`

## 常用命令

最常用的几种运行方式如下。

### 默认运行

```bash
./run.sh
```

这条命令现在会默认同时打开 MuJoCo 和 Rerun。
默认采集时长为 `30s`。

### 真实串口模式

```bash
./run.sh real
```

常用参数：

```bash
./run.sh real --port /dev/ttyUSB0 --baudrate 115200
./run.sh real --duration 60
./run.sh real --no-render
./run.sh real --no-spawn-rerun
./run.sh real --stop-at-excitation-start
./run.sh real --control-mode compensate
```

默认的 `collect` 模式会复用仿真中的逐关节参考轨迹和逆动力学 + PD 控制律，并在真机发送前额外保留安全限位整形。运行结束后会输出：

- `results/real_uart_capture.npz`
- `results/real_uart_capture.json`
- `results/real_friction_identification.npz`
- `results/real_friction_identification.json`
- `results/real_friction_identification_summary.json`

如果你显式切到 `compensate` 模式，它会读取 `results/real_friction_identification_summary.json` 里的当前辨识参数，按

```text
tau = fc * tanh(qd / velocity_scale) + fv * qd + offset
```

计算 7 轴摩擦补偿力矩，再按 `[J1..J7]` 顺序发送给下位机。

如果你只想让机械臂先运动到“预期的激励起始位置”，确认起点没问题后就停，可以加：

```bash
./run.sh real --stop-at-excitation-start
```

这个参数只对 `collect` 模式生效。程序会完成起步过渡和稳定保持，然后发送零力矩并退出，不进入后续激励段，也不会继续做真机摩擦辨识。

### 只关闭 MuJoCo 窗口

```bash
./run.sh --no-render
```

### 只关闭 Rerun 窗口

```bash
./run.sh --no-spawn-rerun
```

### 同时关闭两个窗口

```bash
./run.sh --no-render --no-spawn-rerun
```

### 调整采集时长或采样率

```bash
./run.sh --duration 24 --sample-rate 400
```

## 参数说明

`run.sh` 会把参数原样传给摩擦辨识主程序，常用参数有：

- `--duration`
  - 激励与采集总时长，单位秒，默认 `30`
- `--sample-rate`
  - 记录频率，单位 Hz
- `--base-frequency`
  - 激励轨迹的基频
- `--amplitude-scale`
  - 逐关节分段激励时的关节摆幅缩放，实际摆幅会再受关节限位安全边界约束
- `--feedback-scale`
  - 轨迹跟踪时的 PD 反馈比例
- `--render`
  - 打开 MuJoCo 可视化，默认开启
- `--no-render`
  - 关闭 MuJoCo 可视化
- `--spawn-rerun`
  - 打开 Rerun 可视化，默认开启
- `--no-spawn-rerun`
  - 关闭 Rerun 可视化

查看帮助：

```bash
./run.sh --help
```

如果你想直接修改默认参数，而不是每次通过命令行传参，优先改：

```text
friction_identification_core/config.py
```

这个文件里现在分成几组核心配置：

- `RobotModelConfig`
  - 关节名、关节限位、home 位姿、TCP 偏置、摩擦初值等
- `CollectionConfig`
  - 采集时长、采样率、基频、摆幅、反馈比例、可视化开关等
- `SampleFilterConfig`
  - 限位边界筛样阈值、约束力矩阈值、验证集抽样规则
- `FitConfig`
  - 摩擦拟合的速度尺度、正则化、Huber 参数、最小速度阈值
- `RealUartConfig`
  - 真机 UART 发送策略
  - `send_enabled=True` 时发送完整激励
  - `send_enabled=False` 且 `send_bias_compensation_only=True` 时，仅在 `collect` 模式发送重力+科氏补偿
  - 两者都关时只接收、计算和记录，不执行串口发送

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

- 终端能看到轨迹生成、采集进度、样本筛选、拟合进度等过程性输出
- 终端出现 `Friction identification finished.`
- `results/` 下生成了两个结果文件
- 没有出现 `Failed to load MuJoCo` 或 Python 导入报错

默认情况下，你还能看到 MuJoCo 和 Rerun 对应窗口被正常拉起。

## 关节限位如何参与采集轨迹

工程里已经把 7 个关节的限位统一写入 `friction_identification_core/config.py` 的 `RobotModelConfig.joint_limits`，采集时会同时作用在正向仿真模型和逆动力学模型。

当前轨迹生成策略是：

- 先根据每个关节的上下限建立安全工作区，不直接贴着硬限位跑
- 默认预留约 `10%` 行程、且限制在 `0.04rad` 到 `0.12rad` 之间的安全边界
- 如果 `home` 位姿离某一侧限位太近，会把该关节的激励中心挪到安全区中点，保证正反两个方向都有足够运动量
- 采集轨迹采用“逐关节分段激励”：每一段时间只重点激励一个关节，其余关节保持在各自安全中心位
- 采样筛选时还会再剔除距离限位小于 `0.05rad` 的样本，避免限位附近的约束力矩污染摩擦辨识

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
│   ├── run_simulation.py
│   ├── run_real_uart.py
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
./run.sh --no-render --no-spawn-rerun
```

### 4. 结果文件没有生成

优先检查终端最后几行输出，看是否在采集、拟合或写文件阶段报错。

## 想继续看代码时，从哪里开始

推荐阅读顺序：

1. `run.sh`
2. `friction_identification_core/run_simulation.py`
3. `friction_identification_core/mujoco_driver.py`
4. `friction_identification_core/estimator.py`

这样最容易看清“入口 -> 数据采集 -> 参数拟合”的主流程。
