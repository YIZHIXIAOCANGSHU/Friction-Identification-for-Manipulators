# 逐电机摩擦力辨识重构方案

## 1. 背景与目标

### 1.1 当前架构问题

当前系统采用**并行辨识模式**，存在以下问题：

| 问题 | 描述 |
|------|------|
| 耦合干扰 | 7个关节同时运动，关节间存在动力学耦合，影响辨识精度 |
| 可视化过载 | Rerun 面板为 7关节 × 16+指标，窗口过于密集难以观察 |
| 仿真冗余 | 代码中保留了 MuJoCo 仿真相关逻辑，但实际只需真机运行 |
| 调试困难 | 并行模式下难以定位单个关节的问题 |

### 1.2 重构目标

- **逐电机辨识**：每次只激励一个关节，其他关节保持静止，消除耦合干扰
- **简化可视化**：Rerun 只显示当前辨识关节的数据，减少面板数量
- **移除仿真**：删除仿真相关代码，专注真机辨识
- **保持兼容**：辨识结果格式与现有系统兼容

---

## 2. 新架构设计

### 2.1 辨识流程对比

```
当前并行模式:
┌─────────────────────────────────────────────────────┐
│  Batch 1: J1+J2+J3+J4+J5+J6+J7 同时激励 (90s)       │
├─────────────────────────────────────────────────────┤
│  Batch 2: J1+J2+J3+J4+J5+J6+J7 同时激励 (90s)       │
├─────────────────────────────────────────────────────┤
│  ...重复 N 批次                                      │
└─────────────────────────────────────────────────────┘

新逐电机模式:
┌─────────────────────────────────────────────────────┐
│  Group 1: 逐个关节辨识 (约 7 分钟)                   │
│  ├─ J1 单独激励 (60s) → 辨识 → 保存                  │
│  ├─ J2 单独激励 (60s) → 辨识 → 保存                  │
│  ├─ J3 单独激励 (60s) → 辨识 → 保存                  │
│  ├─ J4 单独激励 (60s) → 辨识 → 保存                  │
│  ├─ J5 单独激励 (60s) → 辨识 → 保存                  │
│  ├─ J6 单独激励 (60s) → 辨识 → 保存                  │
│  └─ J7 单独激励 (60s) → 辨识 → 保存                  │
├─────────────────────────────────────────────────────┤
│  Group 2: 重复逐个关节辨识 (可选多组)                │
└─────────────────────────────────────────────────────┘
```

### 2.2 核心概念变更

| 概念 | 当前 | 重构后 |
|------|------|--------|
| 辨识单元 | Batch（所有关节一批） | JointRun（单关节一次） |
| 分组方式 | 按批次分组 | 按关节分组，每组包含多轮 |
| 激励轨迹 | 多关节相位偏移复合轨迹 | 单关节全范围激励轨迹 |
| 可视化 | 7关节全量面板 | 当前关节专注面板 |

### 2.3 新配置结构

```yaml
# default.yaml 新增/修改配置

identification:
  mode: "sequential"  # 新增: "sequential" | "parallel"
  active_joints: [0, 1, 2, 3, 4, 5, 6]
  
  # 逐电机模式专用配置
  sequential:
    joint_duration: 60.0        # 每个关节的激励时长（精细模式）
    inter_joint_delay: 5.0      # 关节间切换等待时间
    num_groups: 3               # 重复辨识组数
    inter_group_delay: 30.0     # 组间等待时间
    
  excitation:
    profile: "single_joint_sweep"  # 新增单关节激励模式
    # ... 其他参数保持
    
visualization:
  spawn_rerun: true
  rerun_mode: "focused"  # 新增: "focused" | "full"
  # focused 模式只显示当前辨识关节
```

---

## 3. 模块重构详情

### 3.1 配置模块 `config.py`

**新增数据类：**

```python
@dataclass(frozen=True)
class SequentialConfig:
    joint_duration: float       # 单关节激励时长
    inter_joint_delay: float    # 关节间等待
    num_groups: int             # 重复组数
    inter_group_delay: float    # 组间等待
```

**修改 `IdentificationConfig`：**
- 新增 `mode: str` 字段（"sequential" | "parallel"）
- 新增 `sequential: SequentialConfig` 字段

### 3.2 轨迹模块 `trajectory.py`

**新增函数：**

```python
def generate_single_joint_excitation(
    *,
    joint_index: int,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    torque_limits: np.ndarray,
    safety_margin: float,
    duration: float,
    sample_rate: float,
    sweep_cycles: int,
) -> ReferenceTrajectory:
    """
    为单个关节生成全范围激励轨迹。
    其他关节保持在 home_qpos 位置不动。
    """
```

**轨迹特点：**
- 只有目标关节有运动指令
- 其他关节 q_cmd = home_qpos，qd_cmd = 0，qdd_cmd = 0
- 激励模式：正弦扫频 + 速度递增 + 零速穿越

### 3.3 Pipeline 模块 `pipeline.py`

**新增类：**

```python
@dataclass(frozen=True)
class JointRunArtifact:
    """单关节单次辨识结果"""
    joint_index: int
    joint_name: str
    group_index: int
    data: CollectedData
    identification: FrictionIdentificationResult | None
    paths: ResultPaths


@dataclass(frozen=True)
class SequentialPipelineResult:
    """逐电机辨识总结果"""
    source: str
    mode: str
    joint_runs: tuple[JointRunArtifact, ...]
    summary_paths: ResultPaths | None


class SequentialIdentificationPipeline:
    """逐电机摩擦力辨识流水线"""
    
    def run(self) -> SequentialPipelineResult:
        results = []
        for group_idx in range(num_groups):
            for joint_idx in active_joints:
                # 1. 生成单关节轨迹
                # 2. 采集数据
                # 3. 辨识该关节
                # 4. 保存结果
                # 5. 更新可视化
                # 6. 等待冷却
        return SequentialPipelineResult(...)
```

### 3.4 数据源模块 `sources/hardware.py`

**修改 `HardwareSource.collect()`：**

- 新增参数 `target_joint_index: int | None`
- 当 `target_joint_index` 指定时，只对该关节进行激励
- 其他关节发送零力矩

**核心优化 1：简化串口接收逻辑**

采用更简洁的帧解析方式，参考独立监控脚本的实现：

```python
# 帧格式定义
FRAME_HEAD = 0xA5
FRAME_FORMAT = '<BBBfffff'  # head, motor_id, state, pos, vel, tor, t_mos, t_coil
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)

# 简化的接收循环
buffer = bytearray()
latest_positions = [0.0] * 7
latest_velocities = [0.0] * 7
latest_torques = [0.0] * 7

while True:
    # 读取所有可用数据
    if ser.in_waiting > 0:
        buffer.extend(ser.read(ser.in_waiting))
    
    # 解析完整帧
    while len(buffer) >= FRAME_SIZE:
        if buffer[0] == FRAME_HEAD:
            frame_data = buffer[:FRAME_SIZE]
            parsed = struct.unpack(FRAME_FORMAT, frame_data)
            _, motor_id, state, pos, vel, tor, t_mos, t_coil = parsed
            
            # 更新对应电机的数据
            if 1 <= motor_id <= 7:
                idx = motor_id - 1
                latest_positions[idx] = pos
                latest_velocities[idx] = vel
                latest_torques[idx] = tor
            
            buffer = buffer[FRAME_SIZE:]
        else:
            buffer.pop(0)  # 跳过无效字节，寻找帧头
    
    # 只要目标关节有数据就可以推进控制
    target_idx = target_joint_index
    if latest_positions[target_idx] is not None:
        # 执行控制逻辑...
```

**优势：**
- 移除复杂的 `FeedbackCycleWindow` 和 `SerialFrameReader` 类
- 不再需要等待所有关节反馈都刷新
- 只关注当前激励关节的数据
- 代码更简洁，调试更容易

**力矩下发协议：**

```python
# 力矩命令帧格式
# 0xAA 0x55 + tau1..tau7 (little-endian float32) + XOR校验 + 0x55 0xAA

SEND_HEAD = bytes([0xAA, 0x55])
SEND_TAIL = bytes([0x55, 0xAA])

def pack_torque_command(tau: np.ndarray) -> bytes:
    """
    打包7轴力矩命令帧。
    
    帧结构:
    - 帧头: 0xAA 0x55 (2字节)
    - 力矩: tau1..tau7, little-endian float32 (28字节)
    - 校验: XOR (1字节)
    - 帧尾: 0x55 0xAA (2字节)
    总长度: 33字节
    """
    tau = np.asarray(tau, dtype=np.float32).reshape(7)
    payload = struct.pack('<7f', *tau)
    
    # XOR 校验：对 payload 所有字节异或
    xor_check = 0
    for byte in payload:
        xor_check ^= byte
    
    return SEND_HEAD + payload + bytes([xor_check]) + SEND_TAIL

# 使用示例
tau_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
tau_cmd[target_joint_index] = computed_torque  # 只对目标关节施加力矩
frame = pack_torque_command(tau_cmd)
ser.write(frame)
```

**核心优化 2：激励前归零位**

每个关节激励开始前，先将电机移动到位置 0 点（home_qpos），确保从一致的起始位置开始激励：

```python
def _move_to_zero_position(
    self,
    joint_index: int,
    duration_s: float = 3.0,
) -> None:
    """
    激励前将目标关节移动到 0 点位置。
    使用五次多项式轨迹平滑过渡，避免冲击。
    """
    # 1. 读取当前位置
    current_q = self._read_current_position(joint_index)
    target_q = 0.0  # home_qpos[joint_index]
    
    # 2. 生成平滑过渡轨迹
    trajectory = build_quintic_point_to_point_trajectory(
        start_q=current_q,
        goal_q=target_q,
        duration=duration_s,
        sample_rate=self.config.sampling.rate,
    )
    
    # 3. 执行归零运动
    for sample in trajectory:
        tau_cmd = controller.compute_torque(...)
        ser.write(frame_packer.pack(tau_cmd))
```

流程变为：
```
归零J1(3s) → 激励J1(60s) → 归零J2(3s) → 激励J2(60s) → ...
```

**配置参数：**
```yaml
sequential:
  joint_duration: 60.0        # 每个关节的激励时长
  zero_position_duration: 3.0 # 归零运动时长
  inter_joint_delay: 2.0      # 关节间等待时间（归零后稳定）
```

**新增方法：**

```python
def collect_single_joint(
    self,
    *,
    joint_index: int,
    reference: ReferenceTrajectory,
    controller: FrictionIdentificationController,
    safety: SafetyGuard,
    group_index: int,
    total_groups: int,
) -> CollectedData:
    """采集单关节辨识数据，只等待目标关节反馈"""
```

### 3.5 可视化模块 `visualization.py`

**新增类：**

```python
class FocusedJointReporter:
    """单关节专注可视化报告器"""
    
    def __init__(
        self,
        *,
        app_name: str,
        joint_names: list[str],
        spawn: bool = True,
    ) -> None:
        # 初始化时不创建所有关节面板
        
    def set_focus_joint(self, joint_index: int) -> None:
        """切换当前关注的关节，重建面板布局"""
        
    def log_step(self, ...) -> None:
        """只记录当前关注关节的数据"""
```

**面板布局（focused 模式）：**

```
┌─────────────────────────────────────────────────────┐
│  Overview Tab                                        │
│  ┌─────────────┬─────────────┬─────────────────────┐│
│  │ Runtime     │ Current     │ Progress            ││
│  │ Status      │ Joint Info  │ J1✓ J2✓ J3→ J4 ... ││
│  └─────────────┴─────────────┴─────────────────────┘│
│  ┌─────────────────────────────────────────────────┐│
│  │ Position & Velocity (当前关节)                   ││
│  └─────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────┐│
│  │ Torque (当前关节)                                ││
│  └─────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────┐│
│  │ Identification Result (当前关节)                 ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### 3.6 结果存储 `results.py`

**目录结构变更：**

```
results/
├── runs/
│   └── 20260417_120000_sequential/
│       ├── run_manifest.json
│       ├── group_01/
│       │   ├── joint_0_capture.npz
│       │   ├── joint_0_identification.npz
│       │   ├── joint_1_capture.npz
│       │   ├── joint_1_identification.npz
│       │   └── ...
│       ├── group_02/
│       │   └── ...
│       └── summary/
│           ├── hardware_identification_summary.npz
│           └── hardware_identification_report.md
```

### 3.7 入口模块 `__main__.py`

**CLI 参数变更：**

```python
parser.add_argument(
    "--mode",
    choices=("collect", "sequential", "compensate", "compare"),
    default="sequential",  # 默认改为逐电机模式
    help="sequential: 逐电机辨识, collect: 并行辨识(旧), compensate: 补偿验证",
)

parser.add_argument(
    "--joints",
    type=str,
    default=None,
    help="指定辨识的关节，如 '0,2,4' 或 'all'",
)
```

---

## 4. 移除仿真相关代码

### 4.1 需要清理的模块

| 文件 | 清理内容 |
|------|----------|
| `mujoco_env.py` | 保留逆动力学计算，移除仿真循环 |
| `mujoco_support.py` | 保留模型加载，移除仿真辅助 |
| `visualization.py` | 移除 `PoseEstimator` 的 viewer 渲染 |
| `config.py` | `visualization.render` 默认改为 False |

### 4.2 保留的 MuJoCo 功能

- 逆动力学计算（`RigidBodyDynamics.inverse_dynamics()`）
- 正运动学计算（末端位姿估计）
- 模型加载和关节映射

---

## 5. 实现优先级

### Phase 1: 核心流程（必须）

1. 新增 `SequentialConfig` 配置
2. 实现 `generate_single_joint_excitation()` 轨迹生成
3. 实现 `SequentialIdentificationPipeline` 流水线
4. 修改 `HardwareSource` 支持单关节采集
5. 更新 CLI 入口

### Phase 2: 可视化优化（重要）

1. 实现 `FocusedJointReporter` 
2. 添加辨识进度面板
3. 简化面板布局

### Phase 3: 清理与优化（可选）

1. 移除仿真相关代码
2. 优化结果存储结构
3. 添加断点续传功能

---

## 6. 风险与注意事项

### 6.1 安全考虑

- 单关节运动时，其他关节需要保持力矩以抵抗重力
- 关节切换时需要平滑过渡，避免冲击
- 保留现有的关节限位和力矩限制逻辑

### 6.2 兼容性

- 辨识结果格式保持兼容，补偿模式无需修改
- 保留并行模式作为可选项（`--mode collect`）
- 历史结果文件可继续使用

### 6.3 性能

- 逐电机模式总时长：7关节 × 60s × 3组 ≈ 21分钟（不含等待）
- 相比并行模式（90s × 5批 = 7.5分钟）更长，但辨识质量显著提高
- 单组运行约 7 分钟，适合快速验证

---

## 7. 验收标准

- [ ] 能够逐个关节进行辨识，其他关节保持静止
- [ ] Rerun 可视化只显示当前辨识关节，面板数量大幅减少
- [ ] 辨识结果与现有补偿模式兼容
- [ ] 支持指定部分关节进行辨识
- [ ] 支持中断后从断点继续
