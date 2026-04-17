# 摩擦力辨识"等待新鲜快照"卡住问题排查与解决

## 问题现象

运行 `collect` 模式时，系统持续输出：
```
[INFO] 等待 active_joints 形成完整新鲜快照，继续发送零力矩保持安全。
covered=[], progress=0/7, initialized=[1,2,3,4,5,6,7], fresh=[1,2,3,4,5,6,7], 
pending=[1,2,3,4,5,6,7], missing=[], stale=[]
```

关键观察：
- `initialized` 和 `fresh` 都包含全部7个关节 → 串口通信正常，数据在持续接收
- `covered=[]` 且 `progress=0/7` → `FeedbackCycleWindow` 没有记录任何关节
- `pending=[1,2,3,4,5,6,7]` → 窗口认为所有关节都还在等待

## 根因分析

问题出在 [`FeedbackCycleWindow.push()`](friction_identification_core/sources/hardware.py:112) 方法的过滤逻辑：

```python
def push(self, joint_index: int) -> None:
    if not self.active_joint_mask[joint_index]:  # 关键检查
        return
    if self._seen[joint_index]:
        return
    self._window.append(joint_index)
    self._seen[joint_index] = True
```

`active_joint_mask` 由配置文件 [`default.yaml`](friction_identification_core/default.yaml:25) 中的 `identification.active_joints: [0,1,2,3,4,5,6]` 生成。

**可能原因：**

1. **配置与实际关节ID不匹配**：配置使用 0-indexed（0-6），但串口协议可能返回 1-indexed（1-7）的 `motor_id`
2. **active_joint_mask 生成错误**：mask 数组可能全为 False
3. **串口帧解析问题**：`frame.motor_id` 可能超出预期范围

## 解决方案

### 方案一：检查配置文件（推荐先尝试）

确认 [`default.yaml`](friction_identification_core/default.yaml) 中的配置：

```yaml
identification:
  active_joints: [0, 1, 2, 3, 4, 5, 6]  # 确保是 0-indexed
```

如果你的电机ID是 1-7，配置应该保持 0-6（代码内部会转换）。

### 方案二：验证 active_joint_mask 生成

在 [`config.py`](friction_identification_core/config.py) 中查找 `active_joint_mask` 的生成逻辑，确认：
- 数组长度等于 `joint_count`（7）
- 对应索引位置为 True

### 方案三：添加调试日志

在 [`hardware.py`](friction_identification_core/sources/hardware.py:498) 的 `push` 调用前添加临时日志：

```python
# 在 feedback_window.push(idx) 之前添加
print(f"DEBUG: motor_id={frame.motor_id}, idx={idx}, mask[idx]={active_joint_mask[idx]}")
```

这将帮助确认：
- `idx` 的实际值（应为 0-6）
- `active_joint_mask[idx]` 是否为 True

### 方案四：检查串口协议

查看 [`serial_protocol.py`](friction_identification_core/serial_protocol.py) 确认 `motor_id` 的解析是否正确。电机返回的ID可能需要减1才能匹配 0-indexed 的 mask。

## 快速验证步骤

1. **打印 active_joint_mask**：在程序启动时添加：
   ```python
   print(f"active_joint_mask = {active_joint_mask}")
   print(f"active_joint_indices = {active_joint_indices}")
   ```

2. **检查第一个收到的帧**：
   ```python
   print(f"Received frame: motor_id={frame.motor_id}")
   ```

3. **确认索引转换**：代码中 `idx = frame.motor_id - 1`，如果电机返回 1-7，则 idx 为 0-6，这是正确的。

## 最可能的修复

根据日志显示 `initialized` 和 `fresh` 都正确包含了所有关节，说明数据确实在接收。问题很可能是：

**`FeedbackCycleWindow` 在每次循环后被 `advance_after_emit()` 清空，但由于从未触发 `feedback_group_ready`，导致窗口状态一直被重置。**

检查 [`hardware.py:519-526`](friction_identification_core/sources/hardware.py:519) 的条件判断：

```python
feedback_snapshot_ready = (
    feedback_frames_since_emit > 0
    and len(fresh_joint_ids) == len(active_joint_ids)
)
```

如果 `feedback_frames_since_emit` 始终为 0，则永远不会触发。这个计数器只在 `active_joint_mask[idx]` 为 True 时递增。

**结论：问题根源是 `active_joint_mask` 对应位置为 False，导致 `push()` 被跳过，`feedback_frames_since_emit` 不递增。**

## 建议的代码修改

在 [`hardware.py`](friction_identification_core/sources/hardware.py) 第 416-418 行后添加验证：

```python
active_joint_mask = self.config.active_joint_mask
active_joint_indices = np.flatnonzero(active_joint_mask)
active_joint_ids = [int(idx) + 1 for idx in active_joint_indices]

# 添加验证
log_info(f"active_joint_mask: {active_joint_mask.tolist()}")
log_info(f"active_joint_indices: {active_joint_indices.tolist()}")
if not np.any(active_joint_mask):
    raise RuntimeError("active_joint_mask 全为 False，请检查配置文件中的 identification.active_joints")
```

这样可以在启动时立即发现配置问题。
