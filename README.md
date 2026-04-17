# OpenArm 7轴并行摩擦力辨识

这个仓库当前已经收敛为 `真机专用` 的 7 轴并行摩擦力辨识链路，目标很明确：

- 7 个关节同时执行全范围复合激励
- 采集阶段并行控制、并行采样、并行辨识
- 补偿验证阶段只发送摩擦补偿力矩
- `rerun` 实时展示运动、力矩、状态和辨识结果
- 每次运行自动归档，支持跨多次启动做结果对比

## 给新 AI 的最快入口

如果是第一次接手，建议按下面顺序读，通常 10 分钟内就能建立主线：

1. `README.md`
   - 先明确项目边界、运行模式、输出物和 `rerun` 页面结构。
2. `friction_identification_core/default.yaml`
   - 看默认 7 轴配置、批次数、控制器参数、安全窗口和输出文件命名。
3. `friction_identification_core/pipeline.py`
   - 看 collect / compensate 两条主流程如何编排。
4. `friction_identification_core/sources/hardware.py`
   - 这是最重要的真机入口，负责串口采集、控制闭环、在线 `rerun`、离线辨识输入整理。
5. `friction_identification_core/visualization.py`
   - 看 `rerun` 蓝图、实时曲线和辨识结果小图是怎么组织的。
6. 按任务补读
   - 轨迹改动看 `trajectory.py`
   - 控制/安全改动看 `controller.py`
   - 结果文件和 summary 改动看 `results.py`
   - 派生状态和文本摘要看 `status.py`

## 当前系统边界

当前代码默认遵循下面这些前提：

- `hardware only`，不再维护仿真主链路
- 默认激活 7 个关节并行辨识
- 当前有三个运行模式：`collect`、`compensate`、`compare`
- `rerun` 是主调试入口，MuJoCo 只负责姿态/末端位姿可视化

已经移除或不再作为主路径维护的内容：

- 单关节 `target_joint`
- 仿真入口
- `full_feedforward`
- 激励幅值 `amplitude_scale`
- 基础频率 `base_frequency`

## 主流程

### 1. collect

`collect` 会完成一整轮并行采集与辨识：

1. 读取配置，初始化真机源、控制器、安全器、`rerun` 和 MuJoCo 姿态显示
2. 上电后先回到 `robot.home_qpos`，默认就是全 0 位
3. 按 `identification.excitation.window_mode` 生成 7 轴并行激励参考轨迹；默认 `unbounded` 模式不会做位置范围裁切，而是直接做空电机速度激励
4. 真机闭环运行，实时记录 `q / qd / q_cmd / qd_cmd / torque / 温度 / UART`
5. 采集后做速度滤波、刚体逆动力学，并按同一窗口模式筛样
6. 对活跃关节独立拟合 `coulomb / viscous / offset`
7. 保存批次结果，并在所有批次结束后输出 summary / report

### 2. compensate

`compensate` 只做摩擦补偿验证：

1. 读取已有 summary 中的摩擦参数
2. 真机运行补偿模式
3. 记录补偿期的关节状态和力矩状态
4. 输出补偿验证数据文件

### 3. compare

`compare` 会读取历史归档的 collect 结果，生成跨运行对比报告：

1. 扫描 `results/runs/` 下的历史 collect 归档
2. 读取每次运行的 summary
3. 生成 Markdown 和 CSV 对比报告
4. 默认比较最近 5 次，可通过 `--compare-all` 比较全部归档

## 代码地图

- `friction_identification_core/__main__.py`
  - CLI 入口，只负责参数解析和调度 `pipeline`
- `friction_identification_core/default.yaml`
  - 默认配置，建议先看
- `friction_identification_core/pipeline.py`
  - 主编排层，定义 collect / compensate 的批次逻辑
- `friction_identification_core/sources/hardware.py`
  - 真机串口采集、在线控制、实时记录、辨识输入准备
- `friction_identification_core/controller.py`
  - 跟踪控制、摩擦补偿、安全限幅
- `friction_identification_core/trajectory.py`
  - 全范围复合激励轨迹和启动过渡
- `friction_identification_core/visualization.py`
  - `rerun` 蓝图、实时曲线、辨识参数小图
- `friction_identification_core/results.py`
  - 批次数据、summary、Markdown 报告、legacy JSON
- `friction_identification_core/status.py`
  - `rotation_state / range_ratio / limit_margin` 和文本摘要
- `friction_identification_core/mujoco_env.py`
  - MuJoCo 环境和参考轨迹接口

## Rerun 页面约定

当前 `rerun` 主要分成 6 个页面：

- `Overview`
  - 状态文本、辨识摘要、末端位姿，以及几个快速面板
- `By Joint`
  - 每个关节一个单独页签，把位置、速度、命令、力矩、温度、辨识参数都拆成小图
- `Joint Motion`
  - 按运动类参数看全关节对比
- `Torque`
  - 按力矩类参数看全关节对比
- `Identification`
  - 汇总柱状图 + 每个关节每个参数的单独小图
- `Runtime`
  - UART、有效样本比、温度、文本状态和 MuJoCo 姿态

辨识参数小图的约定：

- 路径在 `identification/history/<metric>/Jx`
- 横轴是 `identification_batch`
- 如果只有单批次结果，小图会显示一个点
- 如果有多批次 summary，小图会显示该参数随批次的变化

## 运行方式

交互入口：

```bash
./run.sh
```

快捷入口：

```bash
./run.sh collect
./run.sh compensate
./run.sh compare
```

底层 CLI：

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode collect
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compensate
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compare
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compare --compare-all
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode collect --output results/debug
```

## 输出文件

默认输出目录由 `output.results_dir` 指定，当前默认是 `results/`。

当前输出分成两层：

- `results/`
  - 保存“最新一次”的便捷入口，方便补偿模式直接读取
- `results/runs/<timestamp>_<mode>/`
  - 保存每次启动的独立归档，便于回看和 compare
- `results/comparisons/`
  - 保存跨运行对比生成的 Markdown / CSV

核心产物：

- `hardware_capture_batch_01.npz`
- `hardware_capture_batch_02.npz`
- `hardware_identification_batch_01.npz`
- `hardware_identification_summary.npz`
- `hardware_identification_summary.csv`
- `hardware_identification_report.md`
- `hardware_compensation_validation.npz`
- `real_friction_identification_summary.json`
  - legacy 汇总文件，方便兼容旧流程
- `runs/20260417_153000_collect/`
  - 某次 collect 的完整归档目录
- `comparisons/identification_compare_latest.md`
- `comparisons/identification_compare_latest.csv`

summary 中最关键的数据包括：

- `coulomb / viscous / offset`
- `validation_rmse / validation_r2`
- `valid_sample_ratio / sample_count`
- `batch_coulomb / batch_viscous / batch_offset`
- `batch_validation_rmse / batch_validation_r2`

可直接阅读的文件优先级建议：

1. `hardware_identification_report.md`
2. `hardware_identification_summary.csv`
3. `real_friction_identification_summary.json`

## 常见改动入口

如果后面要继续让 AI 改项目，直接按任务把入口告诉它会最快：

- 改激励覆盖范围或时序：`friction_identification_core/trajectory.py`
- 改 collect 使用安全窗口、物理窗口还是无限位空电机模式：`friction_identification_core/default.yaml` 里的 `identification.excitation.window_mode`
- 改控制律、补偿力矩或安全限幅：`friction_identification_core/controller.py`
- 改串口采集节奏、真机记录字段或辨识筛样：`friction_identification_core/sources/hardware.py`
- 改 `rerun` 页面布局或新增曲线：`friction_identification_core/visualization.py`
- 改 summary 文件结构、批次统计或报告格式：`friction_identification_core/results.py`
- 改文本摘要和旋转状态判定：`friction_identification_core/status.py`

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
