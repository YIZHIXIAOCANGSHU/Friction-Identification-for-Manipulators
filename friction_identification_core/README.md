# Friction Identification Core

这个目录就是当前真机 7 轴并行摩擦力辨识的核心代码。

如果你已经读过仓库根目录 `README.md`，这里建议只把它当成模块内速查表。

## 模块阅读顺序

1. `default.yaml`
   - 看默认配置和输出命名
2. `__main__.py`
   - 看 CLI 如何进入主流程
3. `pipeline.py`
   - 看 collect / compensate 如何组织
4. `sources/hardware.py`
   - 看真机采集、闭环控制、在线可视化和辨识输入整理
5. 按需求补读其他模块

## 文件职责

- `__main__.py`
  - CLI 入口
- `config.py`
  - YAML 解析和配置对象
- `default.yaml`
  - 默认 7 轴并行配置
- `pipeline.py`
  - 批次编排与结果保存调度
- `sources/hardware.py`
  - 真机串口采集、在线控制、在线 `rerun`
- `controller.py`
  - 跟踪控制、摩擦补偿、安全约束
- `trajectory.py`
  - 全范围复合激励和启动过渡
- `visualization.py`
  - `rerun` 蓝图、主页面、按关节页、小图矩阵
- `results.py`
  - 批次结果、summary、Markdown 报告
- `status.py`
  - 旋转状态、范围比、安全裕量、文本摘要
- `mujoco_env.py`
  - MuJoCo 环境和参考轨迹支持

## 常见修改入口

- 想改轨迹覆盖、错相、速度计划：`trajectory.py`
- 想切换 collect 用安全窗口、物理窗口还是无限位空电机模式：`default.yaml` 里的 `identification.excitation.window_mode`
- 想改控制器或补偿力矩：`controller.py`
- 想改 `rerun` 页面布局或新增监控量：`visualization.py`
- 想改 summary 结构或报告内容：`results.py`
- 想改筛样逻辑或真机数据记录：`sources/hardware.py`

## 当前约定

- 只维护真机主链路
- 默认所有 7 个关节并行激活
- `collect` 上电后默认先回 `robot.home_qpos`，当前默认就是全 0 位
- 默认 `collect` 使用 `window_mode: unbounded` 做无限位空电机速度激励和筛样
- `collect` 输出批次数据和 summary
- `compensate` 从 summary 中读取摩擦参数做补偿验证
- `compare` 读取 `results/runs/` 的历史 collect 归档并生成对比报告
