# 7轴并行摩擦力辨识真机改造计划

## 1. 本次计划的硬约束

- 只保留真机链路，不再保留仿真入口、仿真结果、仿真可视化。
- 7 个电机同时参与采集与辨识，但参数仍然按电机独立拟合。
- 激励不再使用“激励幅值”作为配置项，改为基于安全关节窗口的全范围覆盖式复杂激励。
- 补偿验证阶段实际发送的力矩只允许是电机摩擦补偿力矩，不再发送 full feedforward 力矩。
- 必要运行参数必须通过 `rerun` 实时显示。
- `rerun` 中必须明确可视化每个电机当前的旋转状态。
- 新计划直接覆盖当前 `parallel_friction_identification_plan.md`。

---

## 2. 改造目标

### 2.1 核心目标

- 将当前单关节辨识流程改为 7 轴并行采集、并行辨识、并行补偿验证。
- 提升辨识可行性，重点增强低速段、换向段、中高速段、全行程段的数据覆盖。
- 明确区分采集力矩、摩擦残差力矩、摩擦补偿力矩，避免验证阶段混入额外动力学前馈。
- 用 `rerun` 统一展示运行状态、辨识状态、安全状态、补偿状态。

### 2.2 预期效果

- 同一次采集中覆盖 7 个电机的正转、反转、换向、停顿、全行程区间。
- 减少单一正弦激励对速度覆盖不足、零速附近样本不足、单频率可辨性不足的问题。
- 让补偿验证结果更聚焦于“摩擦模型本身是否有效”，而不是混合了逆动力学前馈的效果。

---

## 3. 总体方案调整

### 3.1 运行模式收敛

保留的模式：

- `collect`：并行采集 + 并行辨识
- `compensate`：仅发送摩擦补偿力矩进行验证

删除的模式：

- `sim`
- `full_feedforward`

### 3.2 识别对象调整

现状：

- `target_joint` 驱动的单关节辨识
- 控制器只给单关节输出非零力矩
- 硬件采集只盯某一个目标关节的新鲜反馈

目标：

- 所有 7 个关节同时激活
- 控制器按 7 轴参考轨迹统一输出力矩
- 硬件采集按“全轴可用”或“全轴最小可用集”判定样本有效性

---

## 4. 激励策略重构

## 4.1 设计原则

- 不再配置每个关节的激励幅值。
- 每个关节的目标运动范围直接由安全关节窗口自动计算。
- 激励必须覆盖：
  - 正向运动
  - 反向运动
  - 零速附近换向
  - 低速段
  - 中速段
  - 高频小扰动段
- 7 个关节不能完全同相运动，需要错相与分段，以降低耦合、碰撞风险和串口负载峰值。

## 4.2 新激励形态

将当前“简单正弦波”改为“全范围复合激励”，每个关节的参考轨迹由以下部分叠加或串接组成：

1. 安全中心对位阶段
2. 全安全窗口往返扫描阶段
3. 换向附近微抖动阶段
4. 多速度等级扫频阶段
5. 收尾回中或保持阶段

建议的单关节轨迹组成：

```text
q_ref(j) =
  full_range_sweep(j)
  + harmonic_dither(j)
  + reversal_window_dither(j)
  + phase_offset(j)
```

其中：

- `full_range_sweep(j)`：覆盖该关节安全上下界之间的大范围往返运动
- `harmonic_dither(j)`：在主运动上叠加较小的复合谐波，提高速度分布丰富度
- `reversal_window_dither(j)`：在换向附近生成额外样本，增强库仑摩擦和零速邻域可辨识性
- `phase_offset(j)`：每个关节使用不同相位和时序，避免 7 轴完全同步

## 4.3 激励配置建议

不再保留 `amplitude_scale` 或按轴 `amplitudes` 配置，改为：

```yaml
identification:
  active_joints: [0, 1, 2, 3, 4, 5, 6]
  excitation:
    profile: "full_range_compound"
    duration: 90.0
    sweep_cycles: 4
    reversal_pause_s: 0.15
    zero_crossing_dither_s: 0.30
    harmonic_weights: [1.0, 0.35, 0.12]
    speed_schedule: [0.15, 0.35, 0.70, 1.00]
    phase_offsets: [0.00, 0.11, 0.27, 0.43, 0.19, 0.36, 0.52]
```

说明：

- 轨迹幅度不再手动配置。
- 每个关节的上下界直接取 `joint_limits` 扣除 `safety.joint_limit_margin` 后的安全窗口。
- `speed_schedule` 控制不同阶段的目标运动速度，而不是角度幅值。
- 若某关节行程较小，也仍然执行其可用安全窗口内的全范围扫描。

## 4.4 轨迹实现方向

`trajectory.py` 中不再以“某个目标关节分段激励”为核心，而是新增：

```python
def generate_parallel_full_range_excitation(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    safety_margin: float,
    duration: float,
    sample_rate: float,
    sweep_cycles: int,
    speed_schedule: np.ndarray,
    phase_offsets: np.ndarray,
    harmonic_weights: np.ndarray,
) -> ReferenceTrajectory:
    ...
```

关键点：

- 输出 `q_cmd / qd_cmd / qdd_cmd` 的 shape 全部为 `[N, 7]`
- 所有关节同时有参考轨迹，不再依赖 `target_joint_mask`
- 全轨迹必须在安全窗口内
- 每个关节至少经历一次完整正向和反向覆盖

---

## 5. 控制与补偿策略调整

## 5.1 采集阶段

`collect` 模式中仍允许使用轨迹跟踪控制，但目标是获得可用于辨识的高质量样本：

- 跟踪控制输出：用于让关节按复杂激励运行
- 实际辨识输入：使用测得关节速度和摩擦残差力矩
- 所有关节一起控制，不再屏蔽非目标关节

采集阶段的关键日志量应拆分为：

- `tau_track_ff`
- `tau_track_fb`
- `tau_command`
- `tau_measured`
- `tau_residual_for_identification`

## 5.2 补偿验证阶段

`compensate` 模式的发送力矩必须严格限定为：

```python
tau_command = clamp(
    soften_near_joint_limits(
        predict_compensation_torque(qd_measured, identified_parameters)
    )
)
```

必须明确删除以下行为：

- 不再发送 `inverse_dynamics + friction_compensation`
- 不再发送 `full_feedforward + feedback`
- 不再复用 `full_feedforward` 模式做补偿验证

## 5.3 电机旋转状态定义

每个电机在运行时需要实时派生一个旋转状态：

- `+1`：正转
- `0`：近似静止或换向窗口
- `-1`：反转

建议判定逻辑：

```python
if qd > velocity_eps:
    rotation_state = +1
elif qd < -velocity_eps:
    rotation_state = -1
else:
    rotation_state = 0
```

这个状态既用于 `rerun` 展示，也用于调试复杂激励是否真正覆盖了双向运动。

---

## 6. Rerun 可视化规划

## 6.1 必须显示的参数

以下参数列为强制接入 `rerun` 的必要参数：

| 类别 | 参数 | 用途 |
|------|------|------|
| 关节状态 | `q` | 查看当前位置 |
| 关节状态 | `qd` | 查看当前速度 |
| 关节状态 | `rotation_state` | 直接判断正转 / 反转 / 停止 |
| 关节状态 | `range_ratio` | 查看当前处于安全窗口的哪个位置 |
| 轨迹状态 | `q_cmd` | 查看命令位置 |
| 轨迹状态 | `qd_cmd` | 查看命令速度 |
| 力矩状态 | `tau_measured` | 查看实测力矩 |
| 力矩状态 | `tau_command` | 查看实际下发力矩 |
| 力矩状态 | `tau_track_ff` | 采集阶段前馈分量 |
| 力矩状态 | `tau_track_fb` | 采集阶段反馈分量 |
| 力矩状态 | `tau_friction_comp` | 补偿模式下的摩擦补偿力矩 |
| 辨识状态 | `tau_residual` | 辨识使用的残差力矩 |
| 安全状态 | `limit_margin_remaining` | 距离关节限位还剩多少 |
| 运行状态 | `batch_index` | 当前批次 |
| 运行状态 | `phase_name` | 当前激励阶段 |
| 健康状态 | `mos_temperature` | 功率侧温度 |
| 健康状态 | `coil_temperature` | 电机侧温度 |
| 通信状态 | `uart_cycle_hz` | 通讯频率 |
| 通信状态 | `uart_latency_ms` | 周期延迟 |
| 通信状态 | `uart_transfer_kbps` | 吞吐率 |

## 6.2 必须新增的 Rerun 视图

建议将 `HardwareRerunReporter` 重构为 4 个主页面：

1. `Joint Motion`
   - 7 个关节位置曲线
   - 7 个关节速度曲线
   - 7 个关节旋转状态曲线
   - 7 个关节安全窗口归一化位置条

2. `Torque`
   - 7 个关节实测力矩
   - 7 个关节下发力矩
   - 7 个关节摩擦补偿力矩
   - 7 个关节辨识残差力矩

3. `Runtime`
   - UART 频率、延迟、吞吐
   - 温度
   - 当前批次、当前阶段、有效样本比
   - 末端位姿或整机姿态显示

4. `Identification Summary`
   - 每个关节的 `coulomb`
   - 每个关节的 `viscous`
   - 每个关节的 `offset`
   - 每个关节的 `validation_rmse`
   - 每个关节的 `validation_r2`

## 6.3 电机旋转状态的可视化要求

除了 `q` 与 `qd` 曲线，还要单独显示“当前旋转状态”，不能只靠人眼读速度符号。

建议新增：

- `joint_state/rotation_state/Jx`
- `joint_state/range_ratio/Jx`
- `joint_state/phase/Jx`

并在 `rerun` 中增加一个汇总文本面板，实时输出类似表格：

```text
J1: Forward   | ratio=0.82 | qd=0.41
J2: Reverse   | ratio=0.33 | qd=-0.27
J3: Hold      | ratio=0.51 | qd=0.01
...
```

这样可以直接确认：

- 每个电机是否真的在转
- 转动方向是否符合计划
- 是否已经覆盖到全行程区域
- 是否长时间卡在某一侧或零速附近

---

## 7. 配置与代码改造清单

## 7.1 需要修改的文件

| 路径 | 修改方向 |
|------|----------|
| `friction_identification_core/config.py` | 去掉单关节和仿真导向配置，补充并行激励与批次配置 |
| `friction_identification_core/default.yaml` | 改成真机专用配置，不再包含激励幅值与仿真段 |
| `friction_identification_core/trajectory.py` | 实现全范围复合激励轨迹 |
| `friction_identification_core/controller.py` | 去掉单关节力矩屏蔽，区分采集力矩与摩擦补偿力矩 |
| `friction_identification_core/sources/hardware.py` | 改为 7 轴并行采集、7 轴并行补偿、7 轴状态可视化 |
| `friction_identification_core/pipeline.py` | 改成真机单 pipeline |
| `friction_identification_core/visualization.py` | 增强 `rerun` 面板，加入旋转状态与辨识摘要 |
| `friction_identification_core/results.py` | 保存批次数据、并行辨识结果、补偿验证结果 |
| `friction_identification_core/__main__.py` | 简化为硬件专用 CLI |
| `run.sh` | 删掉仿真菜单、仿真快捷模式、单关节选择交互 |
| `README.md` | 改写为真机并行辨识说明 |

## 7.2 建议新增的文件

| 路径 | 作用 |
|------|------|
| `friction_identification_core/batch_runner.py` | 多批次采集与冷却调度 |
| `friction_identification_core/consistency.py` | 多批次一致性分析 |
| `friction_identification_core/status.py` | 旋转状态、范围比、阶段标签等派生状态计算 |

---

## 8. 明确需要删除的部分

本节是执行时必须逐项核对的删除清单。

## 8.1 配置层删除项

| 标记 | 项目 | 处理 |
|------|------|------|
| `[删除]` | `identification.target_joint` | 不再允许单关节模式 |
| `[删除]` | `identification.excitation.amplitude_scale` | 不再使用激励幅值 |
| `[删除]` | `identification.excitation.base_frequency` | 用复合激励参数替代 |
| `[删除]` | `simulation_friction` | 不再保留仿真摩擦真值 |
| `[删除]` | `output.simulation_results_filename` | 不再输出仿真结果 |
| `[删除]` | `output.simulation_prefix` | 不再使用仿真前缀 |

## 8.2 轨迹与控制删除项

| 标记 | 路径 / 逻辑 | 处理 |
|------|-------------|------|
| `[删除]` | `build_target_joint_mask()` | 不再使用目标关节掩码 |
| `[删除]` | `generate_segmented_excitation_trajectory()` | 不再使用单关节分段激励 |
| `[删除]` | `controller.py` 中按 `target_joint` 清零其余关节力矩 | 全部移除 |
| `[删除]` | `full_feedforward` 补偿路径 | 完全移除 |

## 8.3 真机数据源删除项

| 标记 | 路径 / 逻辑 | 处理 |
|------|-------------|------|
| `[删除]` | `target_joint_idx` 相关逻辑 | 改为全轴并行 |
| `[删除]` | “只等待目标关节反馈”的判定 | 改为全轴反馈健康度判定 |
| `[删除]` | `mode == full_feedforward` 分支 | 删除 |
| `[删除]` | 补偿模式下混入跟踪前馈 / 反馈的逻辑 | 删除 |

## 8.4 仿真链路删除项

| 标记 | 路径 / 逻辑 | 处理 |
|------|-------------|------|
| `[删除]` | `friction_identification_core/sources/simulation.py` | 删除文件 |
| `[删除]` | `pipeline.py::run_simulation()` | 删除 |
| `[删除]` | `__main__.py` 中 `--source sim` | 删除 |
| `[删除]` | `visualization.py::SimulationRerunReporter` | 删除 |
| `[删除]` | `visualization.py::build_simulation_reporter()` | 删除 |
| `[删除]` | `run.sh` 中 `sim` / `sim-ff` | 删除 |
| `[删除]` | `README.md` 中全部仿真运行说明 | 删除 |

## 8.5 CLI 与交互删除项

| 标记 | 项目 | 处理 |
|------|------|------|
| `[删除]` | `--joint` | 不再允许运行时切单轴 |
| `[删除]` | `--source` | CLI 固定走真机 |
| `[删除]` | `--mode full_feedforward` | 删除 |
| `[删除]` | `run.sh` 中“修改目标关节”交互 | 删除 |

---

## 9. 结果与批次规划

## 9.1 批次采集

建议保留多批次采集，以便判断并行辨识是否稳定。

```yaml
batch_collection:
  num_batches: 5
  inter_batch_delay: 30.0
```

每一批次都执行：

1. 复杂全范围并行采集
2. 单批次并行辨识
3. 保存辨识结果
4. 记录批次健康度和样本有效率

## 9.2 输出结果

建议输出收敛为：

```text
results/
├── hardware_capture_batch_01.npz
├── hardware_capture_batch_02.npz
├── hardware_capture_batch_03.npz
├── hardware_capture_batch_04.npz
├── hardware_capture_batch_05.npz
├── hardware_identification_summary.npz
├── hardware_identification_report.md
└── hardware_compensation_validation.npz
```

## 9.3 汇总指标

汇总报告至少包含：

- 每个关节的 `coulomb`
- 每个关节的 `viscous`
- 每个关节的 `offset`
- 每个关节的验证集 `RMSE`
- 每个关节的验证集 `R2`
- 每个关节的多批次一致性
- 每个关节的有效样本占比

---

## 10. 分阶段实施顺序

## 第一阶段：硬件专用化收敛

1. 删除仿真入口、仿真配置、仿真结果命名。
2. 删除 `target_joint` 与单关节 CLI 入口。
3. 删除 `full_feedforward` 模式。

交付标准：

- CLI 只剩真机模式
- 配置中不再出现单关节和仿真字段

## 第二阶段：复杂激励替换

1. 用全范围复合激励替换单正弦激励。
2. 去掉所有激励幅值配置。
3. 让 7 个关节以错相方式并行执行全范围运动。

交付标准：

- 任一批次中每个关节都出现正转、反转、换向
- 轨迹始终处于安全窗口内

## 第三阶段：并行采集与摩擦补偿验证

1. `collect` 改为 7 轴同时采集。
2. `compensate` 改为只发送摩擦补偿力矩。
3. 样本有效性判定从单关节改为多关节。

交付标准：

- 7 轴同时生成有效辨识输入
- 补偿模式下 `tau_command == tau_friction_comp`

## 第四阶段：Rerun 增强

1. 增加必要参数通道。
2. 增加旋转状态显示。
3. 增加批次、阶段、辨识结果摘要面板。

交付标准：

- 可以直接从 `rerun` 看出每个电机当前是否正转 / 反转 / 停止
- 可以直接看到补偿力矩和实测力矩

## 第五阶段：结果汇总与文档清理

1. 增加多批次汇总输出。
2. 增加辨识结果报告。
3. 更新 `README.md` 和 `run.sh`。

交付标准：

- 结果文件和文档都与“真机 7 轴并行辨识”一致

---

## 11. 验收标准

- 代码中不再存在仿真入口、仿真 CLI、仿真可视化入口。
- 代码中不再存在 `target_joint` 驱动的单轴控制主路径。
- 配置中不再存在激励幅值字段。
- 每个关节都能完成安全窗口内的双向全范围运动。
- `compensate` 模式仅发送摩擦补偿力矩。
- `rerun` 中能看到每个电机的旋转状态、位置、速度、补偿力矩、实测力矩、安全裕量。
- 输出结果可以直接支持多批次对比和最终参数汇总。

---

## 12. 本计划对应的实施重点

这次不是在现有计划上小修小补，而是明确转成下面这条主线：

`真机专用` + `7轴并行` + `复杂全范围激励` + `仅摩擦补偿验证` + `Rerun 必要参数可视化` + `仿真链路彻底删除`

后续实际改代码时，优先级按下面顺序执行：

1. 先删仿真和单关节入口
2. 再替换激励与控制主路径
3. 然后补 `rerun` 与结果汇总
4. 最后清理脚本和文档
