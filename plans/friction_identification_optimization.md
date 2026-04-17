# 摩擦力辨识精度优化方案（组合 B）

## 1. 优化目标

通过配置调整 + 信号滤波 + 多频率激励 + 自适应参数，预期提升辨识精度 35-45%。

---

## 2. 实施步骤

### 步骤 1：修改配置文件（5分钟）

```bash
# 备份原配置
cp friction_identification_core/default.yaml friction_identification_core/default.yaml.bak
```

修改 [`default.yaml`](friction_identification_core/default.yaml)：

```yaml
identification:
  excitation:
    duration: 45.0        # 30 → 45，增加激励时长
    base_frequency: 0.10  # 0.12 → 0.10，降低基频获取更多低速样本
    amplitude_scale: 0.25 # 0.22 → 0.25，略微增加幅值

sampling:
  rate: 500.0             # 400 → 500，提高采样率
  timestep: 0.0004        # 0.0005 → 0.0004，相应调整仿真步长

fitting:
  velocity_scale: 0.025   # 微调
  regularization: 1.0e-6  # 增加正则化
  max_iterations: 20      # 增加迭代次数
  huber_delta: 1.5        # 略微放宽 Huber 阈值
  min_velocity_threshold: 0.005  # 降低速度阈值保留更多样本
```

---

### 步骤 2：添加速度信号滤波（15分钟）

修改 [`friction_identification_core/sources/hardware.py`](friction_identification_core/sources/hardware.py)：

**2.1 在文件顶部添加导入**（约第 5 行）：

```python
from scipy.signal import savgol_filter
```

**2.2 修改 `prepare_identification` 方法**（约第 559 行）：

将原来的加速度计算：
```python
gradient_order = 2 if data.sample_count >= 3 else 1
qdd = np.gradient(data.qd, data.time, axis=0, edge_order=gradient_order)
```

替换为：
```python
# 使用 Savitzky-Golay 滤波器平滑速度并计算加速度
sample_rate = 1.0 / np.mean(np.diff(data.time)) if data.time.size > 1 else 500.0
window_length = min(15, max(5, data.sample_count // 4 * 2 + 1))  # 确保为奇数且在合理范围
if window_length % 2 == 0:
    window_length += 1

# 先平滑速度信号
qd_filtered = savgol_filter(data.qd, window_length, 3, axis=0)
# 再计算加速度（同时完成平滑和求导）
qdd = savgol_filter(qd_filtered, window_length, 3, deriv=1, delta=1.0/sample_rate, axis=0)
```

---

### 步骤 3：增强激励轨迹（20分钟）

修改 [`friction_identification_core/trajectory.py`](friction_identification_core/trajectory.py)：

**3.1 找到 `generate_segmented_excitation_trajectory` 函数中的 pattern 生成代码**（约第 252-255 行）：

将原代码：
```python
pattern = envelope * (
    np.sin(omega * local_t)
    + 0.28 * np.sin(harmonic_ratio * omega * local_t + phase_shift)
)
```

替换为：
```python
# 多频率叠加，覆盖更宽速度范围
base_pattern = (
    np.sin(omega * local_t)
    + 0.30 * np.sin(2.1 * omega * local_t + 0.35)
    + 0.18 * np.sin(3.5 * omega * local_t + 0.7)
    + 0.10 * np.sin(5.2 * omega * local_t + 1.1)
)

# 添加线性扫频分量
chirp_freq_start = 0.5 * omega / (2 * np.pi)
chirp_freq_end = 2.0 * omega / (2 * np.pi)
chirp_phase = 2 * np.pi * (
    chirp_freq_start * local_t 
    + 0.5 * (chirp_freq_end - chirp_freq_start) * local_t**2 / segment_duration
)
chirp_pattern = 0.15 * np.sin(chirp_phase)

pattern = envelope * (base_pattern + chirp_pattern)
```

---

### 步骤 4：自适应 velocity_scale 搜索（30分钟）

修改 [`friction_identification_core/estimator.py`](friction_identification_core/estimator.py)：

**4.1 在 `fit_multijoint_friction` 函数中**（约第 267 行），将固定的 `candidate_velocity_scales` 替换为动态生成：

找到：
```python
for joint_idx in range(num_joints):
    if progress_callback is not None:
        progress_callback(joint_idx + 1, num_joints, joint_names[joint_idx])
    best_score = None
    params = None
    for candidate_scale in candidate_velocity_scales:
```

替换为：
```python
for joint_idx in range(num_joints):
    if progress_callback is not None:
        progress_callback(joint_idx + 1, num_joints, joint_names[joint_idx])
    
    # 根据实际速度分布动态确定搜索范围
    joint_velocity = velocity[train_mask, joint_idx]
    speed = np.abs(joint_velocity)
    speed_nonzero = speed[speed > 1e-6]
    
    if speed_nonzero.size > 10:
        v_10 = float(np.percentile(speed_nonzero, 10))
        v_50 = float(np.percentile(speed_nonzero, 50))
        v_90 = float(np.percentile(speed_nonzero, 90))
        
        # 动态生成候选值
        dynamic_scales = [
            v_10 * 0.3, v_10 * 0.5, v_10 * 0.8,
            v_50 * 0.2, v_50 * 0.4,
            v_90 * 0.1, v_90 * 0.2,
        ]
        candidate_velocity_scales_joint = tuple(sorted(set(
            [float(velocity_scale)] + 
            [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12] +
            [max(0.003, min(0.3, s)) for s in dynamic_scales]
        )))
    else:
        candidate_velocity_scales_joint = candidate_velocity_scales
    
    best_score = None
    params = None
    for candidate_scale in candidate_velocity_scales_joint:
```

---

## 3. 验证流程

### 3.1 仿真验证

```bash
# 运行仿真测试
python -m friction_identification_core sim --joint 5

# 查看结果
cat results/friction_identification_joint_5.json
```

### 3.2 对比指标

| 指标 | 优化前基线 | 优化后目标 |
|------|-----------|-----------|
| 训练集 RMSE | - | 下降 30%+ |
| 验证集 RMSE | - | 下降 25%+ |
| 验证集 R² | - | 提升 0.05+ |

### 3.3 真机验证（可选）

```bash
python -m friction_identification_core hw collect --joint 5
```

---

## 4. 回滚策略

如果优化效果不佳：

```bash
# 恢复配置
cp friction_identification_core/default.yaml.bak friction_identification_core/default.yaml

# 使用 git 恢复代码
git checkout friction_identification_core/estimator.py
git checkout friction_identification_core/trajectory.py
git checkout friction_identification_core/sources/hardware.py
```

---

## 5. 修改清单

| 序号 | 文件 | 修改内容 | 预期提升 |
|------|------|----------|----------|
| 1 | default.yaml | 配置参数调整 | 15-20% |
| 2 | hardware.py | Savitzky-Golay 滤波 | 10% |
| 3 | trajectory.py | 多频率激励 + 扫频 | 8% |
| 4 | estimator.py | 自适应 velocity_scale | 5% |

**总预期提升：35-45%**
**实施时间：1-2 小时**
