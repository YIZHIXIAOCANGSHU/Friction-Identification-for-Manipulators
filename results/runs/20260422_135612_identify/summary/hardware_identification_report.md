# Sequential Motor Identification Summary

- run: `20260422_135612_identify`
- groups: `1`
- motor order: `1,2,3,4,5,6,7`

| motor_id | name | conclusion | recommended_for_runtime | status | high_speed_platform_count | high_speed_valid_rmse | saturation_ratio | tracking_error_ratio | valid_rmse |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | motor_01 | reject | false | insufficient_target_frames | 0 | nan | 0.0000 | 0.0000 | nan |
| 2 | motor_02 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |
| 3 | motor_03 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |
| 4 | motor_04 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |
| 5 | motor_05 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |
| 6 | motor_06 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |
| 7 | motor_07 | reject | false | sync_not_acquired | 0 | nan | 0.0000 | 0.0000 | nan |

## Runtime Conclusions

### Motor 01 motor_01

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: insufficient_target_frames`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `64.700000`
- round_total_duration_s: `64.795716`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 02 motor_02

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.022588`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 03 motor_03

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.016304`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 04 motor_04

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.021044`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 05 motor_05

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.028745`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 06 motor_06

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.021445`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

### Motor 07 motor_07

- recommended_for_runtime: `false`
- conclusion_level: `reject`
- conclusion_text: `辨识失败: sync_not_acquired`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.020315`
- core_parameters: `coulomb=nan, viscous=nan, offset=nan, velocity_scale=nan`
- high_speed_platform_count: `0`
- high_speed_valid_rmse: `nan`

## Platform Coverage

### Motor 01 motor_01

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `insufficient_target_frames`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `2897`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `64.700000`
- round_total_duration_s: `64.795716`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 02 motor_02

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.022588`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 03 motor_03

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.016304`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 04 motor_04

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.021044`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 05 motor_05

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.028745`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 06 motor_06

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.021445`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

### Motor 07 motor_07

- train_platforms: `-`
- valid_platforms: `-`
- validation_mode: `train_only`
- validation_reason: `sync_not_acquired`
- recommended_for_runtime: `false`
- conclusion_level: `reject`
- sequence_error_count: `0`
- target_frame_count: `0`
- planned_duration_s: `64.700000`
- actual_capture_duration_s: `0.000000`
- round_total_duration_s: `2.020315`
- saturation_ratio: `0.0000`
- tracking_error_ratio: `0.0000`

