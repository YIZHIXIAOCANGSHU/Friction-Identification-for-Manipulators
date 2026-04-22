# 单电机顺序摩擦辨识

这个仓库当前真实主流程已经收敛为 `单电机顺序辨识`，不是旧文档里描述的“7 轴并行 collect / compensate / compare”。

当前代码的目标很具体：

- 一次只对一个电机做激励、采集和辨识
- 多个电机按配置顺序串行执行
- CLI 主模式是 `identify` 和 `compensate`
- `default` 和 `sequential` 仅作为 `identify` 的兼容别名
- `./run.sh` 是人工使用的交互式数字向导
- `python3 -m friction_identification_core ...` 是脚本化 / 自动化入口
- 轨迹使用 `hold_start -> 正向平台 -> transition_mid_zero -> 反向平台 -> transition_end_zero -> hold_end`
- 辨识优先使用 `steady-state` 平台样本
- 验证优先使用“按速度平台留出”的方式，而不是时间步抽样
- 结果会直接给出 `recommended / caution / reject` 结论

## 真实系统边界

以当前代码为准，真实边界如下：

- CLI 主模式是 `identify` 和 `compensate`
- `default` 和 `sequential` 都会映射到 `identify`
- 主入口是 `friction_identification_core/__main__.py`
- 主流程编排在 `friction_identification_core/pipeline.py`
- 激励轨迹在 `friction_identification_core/trajectory.py`
- 摩擦辨识在 `friction_identification_core/identification.py`
- 结果汇总和报告在 `friction_identification_core/results.py`
- 默认配置在 `friction_identification_core/default.yaml`
- 自动化验证在 `tests/test_sequential_identification.py`

## 和旧说法的冲突点

下面这些内容曾经出现在旧 README 或历史认知里，但已经不是当前真实行为：

- 旧说法：存在 `collect / compensate / compare`
  当前真实代码：CLI 主模式是 `identify / compensate`
- 旧说法：7 轴并行采集、并行辨识
  当前真实代码：一次只激励一个目标电机，其余电机只作为串口序列上下文
- 旧说法：整段复合轨迹直接拟合
  当前真实代码：优先使用恒速平台的 `steady-state` 样本做静态摩擦拟合
- 旧说法：验证主要依赖时间步抽样
  当前真实代码：优先做速度平台留出验证；样本不足时明确降级并记录原因

历史 `results/` 目录里可能仍然保留旧轮次产物，但这些文件不能代表当前代码边界。

## 主流程

一次 round 的真实语义是：

1. 打开串口并清空输入缓存
2. 等待通信同步
   - 目标电机命中足够数量的合法反馈帧后，就允许进入正式采集
   - 不再依赖总线帧顺序，单帧只校验帧头、长度、`motor_id` 范围和数值有限性
   - 同步前不会开始 round 的正式计时，也不会把样本写入 capture
3. 对目标电机发送单电机恒速平台激励
4. 只记录正式采集窗口内的目标电机样本
5. 先做数据质量检查
   - `synced_before_capture`
   - `sequence_error_count / sequence_error_ratio`
   - `target_frame_count / target_frame_ratio`
   - `planned_duration_s / actual_capture_duration_s / round_total_duration_s`
6. 再做 steady-state 样本筛选
7. 用最简单可落地的静态摩擦模型拟合
   - `torque = coulomb * tanh(v / scale) + viscous * v + offset`
8. 优先按速度平台留出做验证
9. 输出 capture、identification、summary、report

## 轨迹语义

当前激励不再把“整段复合速度轨迹直接拟合”作为主路径，而是改成更适合单电机摩擦辨识的最小可用方案：

- `hold_start`
- 多个正向恒速平台
  - 每个平台分成 `settle_forward_xx`
  - 和 `steady_forward_xx`
- `transition_mid_zero`
- 多个反向恒速平台
  - 每个平台分成 `settle_reverse_xx`
  - 和 `steady_reverse_xx`
- `transition_end_zero`
- `hold_end`

当前轨迹不再根据总 `duration` 自动切分速度段，而是直接由 `default.yaml` 里的 `excitation.platforms` 和 `transition_duration` 显式决定。平台支持两种写法，但每个平台必须且只能提供一个字段：

- `speed`
- `speed_ratio`

默认配置已经切到 5 档相对速度平台：

```yaml
excitation:
  sample_rate: 200.0
  hold_start: 1.0
  hold_end: 1.0
  transition_duration: 0.35
  # 每个平台按顺序执行；速度会在正向和反向各走一遍。
  # speed_ratio 是相对比例，实际目标速度 = speed_ratio * max_velocity。
  platforms:
    # 低速段：更容易看到起步附近的库仑摩擦特性。
    - speed_ratio: 0.12
      # 到达目标速度后先等待瞬态衰减。
      settle_duration: 0.30
      # 真正用于稳态辨识的数据采样窗口。
      steady_duration: 0.90
    # 低中速段：补充更宽一点的速度覆盖。
    - speed_ratio: 0.28
      settle_duration: 0.35
      steady_duration: 1.00
    # 中速段：通常是时长和信噪比较均衡的一档。
    - speed_ratio: 0.50
      settle_duration: 0.45
      steady_duration: 1.20
    # 中高速段：帮助区分更明显的速度相关摩擦项。
    - speed_ratio: 0.72
      settle_duration: 0.60
      steady_duration: 1.40
    # 高速段：补足高速度区间信息，通常需要更长稳定时间。
    - speed_ratio: 0.90
      settle_duration: 0.80
      steady_duration: 1.70
```

- `speed` 是绝对速度值
- `speed_ratio` 会在运行时换算成 `speed_ratio * max_velocity`
- `settle_duration` 是达到目标速度后的等待时间，用来避开瞬态，不直接参与稳态拟合
- `steady_duration` 是稳态采样窗口长度，这一段数据会更直接影响辨识结果
- 反向阶段不会单独配置，运行时会按同一组平台自动镜像
- 平台速度如果超过 `0.90 * max_velocity` 会直接报错，不会静默裁剪

## 辨识语义

当前辨识逻辑的重点不是更复杂的模型，而是更严格的数据契约：

- 只优先使用 `steady_*` 平台样本
- 不把 `settle`、`transition`、`hold` 样本混进拟合
- 只保留 `sign(velocity) == sign(velocity_cmd)` 的样本
- 会过滤跟踪误差过大的样本
- 会过滤接近扭矩饱和的样本
- 每个平台过滤后若不足门槛，会整档剔除并记录到 `dropped_platforms`
- 训练加权由“按速度正负平衡”改成“按平台均衡”
- 5 档平台时固定留出第 3 档和第 5 档的正反向平台做验证
- 4 档平台时固定留出第 2 档和第 4 档
- 少于 4 档时会退化为 `train_only`
- 最终结论会直接给出 `recommended / caution / reject`
- 只有 `recommended_for_runtime = true` 的结果才建议进入后续补偿参数库
- `max_sequence_error_ratio` 和顺序字段仍保留在配置与结果里，但现在只是兼容字段，不再阻塞采集或判废

## 补偿语义

`compensate` 模式会复用当前顺序和激励轨迹，但不会再做辨识。它会：

- 默认读取 `results/hardware_identification_summary.npz`
- 只接受 `recommended_for_runtime = true` 且参数有限的目标电机
- 对当前目标电机使用
  - `速度误差反馈`
  - `+ 基于 reference.velocity_cmd 的摩擦前馈`
- 记录速度跟踪结果，重点看 `expected velocity / actual velocity / velocity_error`

补偿模式不会覆盖根目录的辨识 summary，也不会写 `identification.npz`。

## 执行时间为什么会和预期不一样

当前默认配置下，单个电机一轮 `planned_duration_s` 大约是 `64.7s`，不是旧文档里那组更短平台参数对应的 `20.1s`。原因很直接：

- `hold_start = 1.0s`
- 正向 5 档平台总和 = `26.5s`
- `transition_mid_zero = 0.35s`
- 反向 5 档平台总和 = `26.5s`
- `transition_end_zero = 0.35s`
- `hold_end = 1.0s`

也就是说，理论轨迹时长本身就是：

`1.0 + 26.5 + 0.35 + 26.5 + 0.35 + 1.0 = 64.7s`

实际运行时长还会再叠加：

- `sync_wait_duration_s`
- 存盘和离线辨识开销
- 墙钟时间驱动采集带来的轻微偏差

当前停止条件是“墙钟采集时长达到规划时长”，不是“严格采满 `sample_rate * duration` 个目标帧”。所以样本数和总执行时间都不应该按理想采样周期去硬推。

## 运行方式

交互入口：

```bash
./run.sh
```

底层 CLI：

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compensate
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode default
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --motors 1,3,5
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --groups 2
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --output results/debug
```

说明：

- `./run.sh` 现在始终先进入交互式数字菜单
- `./run.sh help` / `./run.sh -h` / `./run.sh --help` 仍然保留
- `./run.sh identify --motors 1,3,5` 这类旧式非交互调用不再支持
- 菜单里显示的是 `identify` 和 `compensate`
- CLI 里仍然保留 `default / sequential`，但它们只会映射到 `identify`

## 输出物

每次运行会在 `results/runs/<timestamp>_<mode>/` 下归档。

- `identify` 会同步写一份最新 summary 到 `results/`
- `compensate` 只写本次运行目录，不会覆盖根目录辨识 summary

核心产物：

- `group_XX/motor_XX/capture.npz`
- `run_manifest.json`
- `<mode>.rrd`

仅 `identify` 额外产出：

- `group_XX/motor_XX/identification.npz`
- `summary/hardware_identification_summary.npz`
- `summary/hardware_identification_summary.csv`
- `summary/hardware_identification_report.md`

当前最关键的结果字段包括：

- 质量指标
  - `sequence_error_count`
  - `sequence_error_ratio`
  - `target_frame_count`
  - `target_frame_ratio`
  - `planned_duration_s`
  - `actual_capture_duration_s`
  - `round_total_duration_s`
  - `synced_before_capture`
  - `saturation_ratio`
  - `tracking_error_ratio`
- 辨识状态
  - `status`
  - `sample_count`
  - `valid_sample_ratio`
  - `recommended_for_runtime`
  - `conclusion_level`
  - `conclusion_text`
- 平台验证
  - `validation_mode`
  - `validation_reason`
  - `train_platforms`
  - `valid_platforms`
  - `high_speed_platform_count`
  - `high_speed_valid_rmse`
- 模型参数
  - `coulomb`
  - `viscous`
  - `offset`
  - `velocity_scale`

建议先按这个顺序看结果：

1. `hardware_identification_report.md`
2. Rerun 的 `summary/conclusions`
3. 某个电机的 `rounds/*/quality/summary`
4. 最后再看 `coulomb / viscous / offset / velocity_scale`

## 建议阅读顺序

如果要继续接手这个仓库，建议按下面顺序读：

1. `friction_identification_core/__main__.py`
2. `friction_identification_core/default.yaml`
3. `friction_identification_core/pipeline.py`
4. `friction_identification_core/trajectory.py`
5. `friction_identification_core/identification.py`
6. `friction_identification_core/results.py`
7. `tests/test_sequential_identification.py`

## 依赖

```bash
pip3 install -r requirements.txt
```
