from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RobotModelConfig:
    """机器人本体与关节物理属性配置。"""

    # URDF 模型路径。通常只有在更换机械臂模型文件时才需要修改。
    urdf_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "am_d02_model" / "urdf" / "AM-D02-AemLURDF0413.urdf"
    )
    # 参与摩擦辨识的关节名称，顺序必须与后续所有关节向量保持一致。
    joint_names: tuple[str, ...] = (
        "ArmLsecond_Joint",
        "ArmLthird_Joint",
        "ArmLfourth_Joint",
        "ArmLfifth_Joint",
        "ArmLsixth_Joint",
        "ArmLsixthoutput_Joint",
        "ArmLseventh_Joint",
    )
    # 各关节力矩上限，用于轨迹生成与仿真控制时的安全约束。
    torque_limits: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 9.0], dtype=np.float64)
    )
    # 各关节位置限位，单位为弧度，按 [lower, upper] 顺序填写。
    joint_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.8300, 0.8290],
                [-0.8190, 0.0000],
                [-1.5920, 1.6810],
                [0.0000, 1.5230],
                [-1.4280, 1.5360],
                [-0.7384, 0.6513],
                [-0.8899, 1.6220],
            ],
            dtype=np.float64,
        )
    )
    # 默认 home 位姿，也是轨迹初始化和安全中心计算的重要参考点。
    home_qpos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    # TCP 相对末端执行器 body 的位置偏置，单位米。
    tcp_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.07, -0.03], dtype=np.float64))
    # MuJoCo / 动力学模型中用作末端参考的 body 名称。
    end_effector_body: str = "tcp"
    # 各关节摩擦损失真值，用于仿真生成带摩擦的数据。
    friction_loss: np.ndarray = field(
        default_factory=lambda: np.array([0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08], dtype=np.float64)
    )
    # 各关节粘性阻尼真值，用于仿真中的速度相关阻力。
    damping: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.45, 0.38, 0.38, 0.28, 0.26, 0.28], dtype=np.float64)
    )


@dataclass
class CollectionConfig:
    """数据采集阶段的轨迹、采样与可视化配置。"""

    # 单次采集总时长，单位秒。
    duration: float = 30.0
    # 结果记录频率，单位 Hz；值越高，数据越密。
    sample_rate: float = 400.0
    # MuJoCo 仿真步长，单位秒；通常保持较小以保证数值稳定。
    timestep: float = 0.0005
    # 激励轨迹基频，决定逐关节摆动的整体节奏。
    base_frequency: float = 0.12
    # 激励幅值缩放系数；实际幅值还会再受关节限位安全边界约束。
    amplitude_scale: float = 0.22
    # 轨迹跟踪反馈比例；越大跟踪越紧，但也可能引入更强控制扰动。
    feedback_scale: float = 0.2
    # 是否打开 MuJoCo 实时渲染窗口。
    render: bool = True
    # 是否同时拉起 Rerun 可视化窗口。
    spawn_rerun: bool = True

    @property
    def realtime(self) -> bool:
        # 当前实现中，只要开启渲染，就按实时模式驱动仿真。
        return self.render


@dataclass
class SampleFilterConfig:
    """样本筛选与验证集抽样配置。"""

    # 距离关节限位的最小安全边界，单位弧度；过近样本会被剔除。
    limit_margin: float = 0.05
    # 接触/约束力矩容忍阈值；超过该值的样本视为可能被额外约束污染。
    constraint_tolerance: float = 0.35
    # 验证集抽样步长，例如 5 表示每 5 个样本取 1 个候选验证点。
    validation_stride: int = 5
    # 验证集前期预热跳过数量，避免轨迹刚启动时的瞬态影响评估。
    validation_warmup_skip: int = 20

    def build_validation_mask(self, num_samples: int) -> np.ndarray:
        # 构造规则抽样的验证集掩码，并显式排除前段预热样本。
        mask = np.zeros(max(int(num_samples), 0), dtype=bool)
        stride = max(int(self.validation_stride), 1)
        if mask.size > 0:
            mask[::stride] = True
            warmup = min(max(int(self.validation_warmup_skip), 0), mask.size)
            mask[:warmup] = False
        return mask


@dataclass
class FitConfig:
    """摩擦参数拟合器配置。"""

    # 速度特征缩放系数，用于改善拟合时不同量纲下的数值条件。
    velocity_scale: float = 0.03
    # L2 正则项，防止矩阵病态或解过度震荡。
    regularization: float = 1e-8
    # 迭代重加权/鲁棒拟合的最大迭代次数。
    max_iterations: int = 16
    # Huber 损失阈值，用于降低异常点对参数估计的影响。
    huber_delta: float = 1.35
    # 最小有效速度阈值；速度过小时通常难以可靠区分摩擦项。
    min_velocity_threshold: float = 0.01


@dataclass
class FrictionIdentificationConfig:
    """摩擦辨识总配置，统一收口各子模块参数。"""

    # 应用名，主要用于日志或可视化标题展示。
    app_name: str = "AM-D02 Friction Identification"
    # 机器人模型与真实物理参数。
    model: RobotModelConfig = field(default_factory=RobotModelConfig)
    # 数据采集相关参数。
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    # 样本筛选与验证集规则。
    sample_filter: SampleFilterConfig = field(default_factory=SampleFilterConfig)
    # 参数拟合器超参数。
    fit: FitConfig = field(default_factory=FitConfig)

    def with_collection_overrides(
        self,
        *,
        duration: float,
        sample_rate: float,
        base_frequency: float,
        amplitude_scale: float,
        feedback_scale: float,
        render: bool,
        spawn_rerun: bool,
    ) -> "FrictionIdentificationConfig":
        # 命令行传参只覆盖采集层配置，其余配置保持默认值不变。
        return replace(
            self,
            collection=replace(
                self.collection,
                duration=duration,
                sample_rate=sample_rate,
                base_frequency=base_frequency,
                amplitude_scale=amplitude_scale,
                feedback_scale=feedback_scale,
                render=render,
                spawn_rerun=spawn_rerun,
            ),
        )


# 项目默认使用的摩擦辨识配置实例。
DEFAULT_FRICTION_CONFIG = FrictionIdentificationConfig()
