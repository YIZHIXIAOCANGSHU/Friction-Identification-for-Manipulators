# 摩擦力辨识运行时

这个仓库当前主流程已经收敛为单电机顺序运行的 `identify / compensate` 两个模式。

- `identify`:
  先做预归零，再执行零中心、位置有界的 `multisine` 位置激励。
  静态辨识窗口只使用 `excitation_cycle_*`，验证语义改为 `velocity-band` 留出。
  同一份采集数据上还会额外产出 LuGre 动态辨识结果。
- `compensate`:
  继续复用静态 summary 的前馈参数做补偿验证，不混入新的绝对式 PD。
- CLI 主模式是 `identify` 和 `compensate`。
- `default` 和 `sequential` 仅作为 `identify` 的兼容别名。

## 配置语义

`excitation` 主路径固定为 `multisine`:

```yaml
excitation:
  sample_rate: 200.0
  curve_type: "multisine"
  hold_start: 1.0
  hold_end: 1.0
  position_limit: 2.5
  velocity_utilization: 1.0
  base_frequency: 0.20
  steady_cycles: 6
  fade_in_cycles: 2
  fade_out_cycles: 2
  harmonic_multipliers: [1, 2, 3, 4, 5, 6]
  harmonic_weights: [1.0, 0.75, 0.55, 0.40, 0.30, 0.25]
```

相位命名固定为:

- `hold_start`
- `fade_in`
- `excitation_cycle_01..N`
- `fade_out`
- `hold_end`

旧平台路径已经退役，不再支持:

- `excitation.platforms`
- `excitation.transition_duration`

`control` 主配置固定为:

- `position_gain`
- `velocity_gain`
- `zeroing_position_gain`
- `zeroing_velocity_gain`
- `max_velocity`
- `max_torque`
- `zeroing_hard_velocity_limit`
- `zeroing_velocity_limit`
- `zeroing_position_tolerance`
- `zeroing_velocity_tolerance`
- `zeroing_required_frames`
- `zeroing_timeout`
- `speed_abort_ratio`
- `zero_target_velocity_threshold`
- `low_speed_abort_limit`

旧兼容字段也已经移除:

- `control.velocity_p_gain`
- `control.torque_limits`

## 运行时行为

- 默认配置只启用 `1~4` 号电机做 `zeroing / identify / compensate`；`5~7` 号电机会继续上报反馈，但不会被纳入本轮下发目标。
- `identify` 每次 run 开始前会先做 `zeroing`，只对本次启用的电机顺序执行。
- `zeroing` 使用独立的 `zeroing_position_gain / zeroing_velocity_gain`，不复用正式采集阶段的温和跟踪增益。
- `zeroing_hard_velocity_limit` 控制零位阶段的硬速度保护；`zeroing_velocity_limit` 只用于判断是否进入 0 点附近的最终锁定窗口。
- 归零成功判定使用最近 5 帧的中值滤波，并带 1.25x 退出带滞回。
- 普通运行阶段的速度保护使用“当前 phase 内按当前位置匹配到的理论速度”来计算动态阈值。
- 所有 abort、sync 失败和人工中断都会先发送一包全零扭矩，再落盘 manifest 和 abort_event。
- 命令行 `--motors` 只会在配置里的 `motors.enabled` 子集内进一步缩小，不会把未启用电机重新加回运行集合。

## 输出文件

`identify` 会写出两组 summary:

- 静态: `hardware_identification_summary.npz/csv/md`
- 动态: `hardware_dynamic_identification_summary.npz/csv/md`

每个电机目录下还会保存:

- `capture.npz`
- `identification.npz`
- `lugre_identification.npz`

`compensate` 会优先读取当前 `results_dir` 中最近一次已完成 `identify` 运行的静态 summary；
如果没有可用的已完成 `identify` summary，就回退到当前 `results_dir` 根目录下的最新 summary 快照。

## 命令行

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compensate
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --motors 1,3,4
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --groups 2
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify --output results/debug
```
