from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MODEL_COEFFICIENT = 0.1
EPSILON = 1e-12


@dataclass(frozen=True)
class FrictionParameters:
    fc: float
    k: float
    fv: float
    fo: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FitResult:
    parameters: FrictionParameters
    metrics: Dict[str, float]
    sample_count: int


def predict_friction(
    velocity: float,
    params: FrictionParameters,
    model_coefficient: float = MODEL_COEFFICIENT,
) -> float:
    return (
        params.fc * math.tanh(model_coefficient * params.k * velocity)
        + params.fv * velocity
        + params.fo
    )


def load_seed_map(path: str) -> Dict[str, FrictionParameters]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    joints = payload.get("joints", payload)
    result: Dict[str, FrictionParameters] = {}
    for joint_name, values in joints.items():
        result[joint_name] = FrictionParameters(
            fc=float(values["fc"]),
            k=float(values["k"]),
            fv=float(values["fv"]),
            fo=float(values["fo"]),
        )
    return result


def load_samples_from_csv(
    path: str,
    *,
    joint_filter: Optional[Iterable[str]] = None,
    default_joint: str = "joint1",
    coriolis_scale: float = 0.0,
) -> Dict[str, List[Dict[str, float]]]:
    wanted = set(joint_filter or [])
    grouped: Dict[str, List[Dict[str, float]]] = {}

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            joint_name = (_pick(row, "joint", "joint_name") or default_joint).strip()
            if wanted and joint_name not in wanted:
                continue

            velocity_text = _pick(row, "velocity", "joint_velocity")
            if velocity_text is None:
                raise ValueError(f"Row {row_number} is missing a velocity column.")
            velocity = _parse_float(velocity_text, row_number, "velocity")

            friction_target = _extract_target(row, row_number, coriolis_scale)
            weight = _parse_optional_float(
                _pick(row, "weight", "sample_weight"),
                row_number,
                "weight",
                default=1.0,
            )

            grouped.setdefault(joint_name, []).append(
                {
                    "velocity": velocity,
                    "friction_torque": friction_target,
                    "weight": weight,
                }
            )

    if not grouped:
        raise ValueError(f"No usable samples were found in {path}.")

    return grouped


def fit_joint_model(
    samples: Sequence[Dict[str, float]],
    *,
    seed: Optional[FrictionParameters] = None,
    seed_regularization: float = 0.0,
    k_bounds: Tuple[float, float] = (0.5, 400.0),
    coarse_candidates: int = 160,
    refinement_rounds: int = 4,
    refinement_candidates: int = 80,
    model_coefficient: float = MODEL_COEFFICIENT,
) -> FitResult:
    if len(samples) < 4:
        raise ValueError("At least four samples are required to fit the friction model.")

    k_min, k_max = k_bounds
    if k_min <= 0.0 or k_max <= k_min:
        raise ValueError("k_bounds must satisfy 0 < min < max.")

    candidate_ks = _build_candidate_ks(seed, k_min, k_max, coarse_candidates)
    best = _evaluate_candidates(
        samples=samples,
        candidate_ks=candidate_ks,
        seed=seed,
        seed_regularization=seed_regularization,
        model_coefficient=model_coefficient,
    )

    for _ in range(refinement_rounds):
        span_low = max(k_min, best.parameters.k / 1.8)
        span_high = min(k_max, best.parameters.k * 1.8)
        if span_high - span_low < 1e-6:
            break

        refined_candidates = _logspace(span_low, span_high, refinement_candidates)
        refined_candidates.append(best.parameters.k)
        best = _evaluate_candidates(
            samples=samples,
            candidate_ks=refined_candidates,
            seed=seed,
            seed_regularization=seed_regularization,
            model_coefficient=model_coefficient,
            incumbent=best,
        )

    best = _golden_section_refine(
        samples=samples,
        incumbent=best,
        seed=seed,
        seed_regularization=seed_regularization,
        k_bounds=(k_min, k_max),
        model_coefficient=model_coefficient,
    )

    return best


def fit_all_joints(
    samples_by_joint: Dict[str, Sequence[Dict[str, float]]],
    *,
    seed_map: Optional[Dict[str, FrictionParameters]] = None,
    seed_regularization: float = 0.0,
    k_bounds: Tuple[float, float] = (0.5, 400.0),
    coarse_candidates: int = 160,
    refinement_rounds: int = 4,
    refinement_candidates: int = 80,
    model_coefficient: float = MODEL_COEFFICIENT,
) -> Dict[str, FitResult]:
    results: Dict[str, FitResult] = {}
    seed_map = seed_map or {}

    for joint_name, samples in samples_by_joint.items():
        seed = _resolve_seed(seed_map, joint_name)
        results[joint_name] = fit_joint_model(
            samples,
            seed=seed,
            seed_regularization=seed_regularization,
            k_bounds=k_bounds,
            coarse_candidates=coarse_candidates,
            refinement_rounds=refinement_rounds,
            refinement_candidates=refinement_candidates,
            model_coefficient=model_coefficient,
        )

    return results


def build_prediction_rows(
    samples_by_joint: Dict[str, Sequence[Dict[str, float]]],
    fit_results: Dict[str, FitResult],
    *,
    model_coefficient: float = MODEL_COEFFICIENT,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for joint_name, samples in samples_by_joint.items():
        params = fit_results[joint_name].parameters
        for sample in samples:
            predicted = predict_friction(
                sample["velocity"],
                params,
                model_coefficient=model_coefficient,
            )
            rows.append(
                {
                    "joint": joint_name,
                    "velocity": sample["velocity"],
                    "friction_torque": sample["friction_torque"],
                    "predicted_friction": predicted,
                    "residual": sample["friction_torque"] - predicted,
                    "weight": sample.get("weight", 1.0),
                }
            )
    return rows


def _extract_target(row: Dict[str, str], row_number: int, coriolis_scale: float) -> float:
    direct_target = _pick(row, "friction_torque", "residual_torque")
    if direct_target is not None:
        return _parse_float(direct_target, row_number, "friction_torque")

    measured = _pick(row, "measured_torque", "torque", "effort")
    if measured is None:
        raise ValueError(
            f"Row {row_number} needs either friction_torque/residual_torque or measured_torque."
        )

    measured_value = _parse_float(measured, row_number, "measured_torque")
    gravity_value = _parse_optional_float(
        _pick(row, "gravity_torque", "gravity"),
        row_number,
        "gravity_torque",
        default=0.0,
    )
    coriolis_value = _parse_optional_float(
        _pick(row, "coriolis_torque", "coriolis"),
        row_number,
        "coriolis_torque",
        default=0.0,
    )
    return measured_value - gravity_value - coriolis_scale * coriolis_value


def _build_candidate_ks(
    seed: Optional[FrictionParameters],
    k_min: float,
    k_max: float,
    coarse_candidates: int,
) -> List[float]:
    candidates = _logspace(k_min, k_max, coarse_candidates)

    if seed is not None:
        local_min = max(k_min, seed.k / 4.0)
        local_max = min(k_max, seed.k * 4.0)
        candidates.extend(_logspace(local_min, local_max, max(40, coarse_candidates // 2)))
        candidates.append(seed.k)

    return sorted(set(round(value, 12) for value in candidates))


def _evaluate_candidates(
    *,
    samples: Sequence[Dict[str, float]],
    candidate_ks: Sequence[float],
    seed: Optional[FrictionParameters],
    seed_regularization: float,
    model_coefficient: float,
    incumbent: Optional[FitResult] = None,
) -> FitResult:
    best_result = incumbent
    best_objective = incumbent.metrics["objective"] if incumbent is not None else math.inf

    for k_value in candidate_ks:
        params, metrics = _fit_linear_terms_for_fixed_k(
            samples=samples,
            fixed_k=k_value,
            seed=seed,
            seed_regularization=seed_regularization,
            model_coefficient=model_coefficient,
        )
        if metrics["objective"] < best_objective:
            best_objective = metrics["objective"]
            best_result = FitResult(
                parameters=params,
                metrics=metrics,
                sample_count=len(samples),
            )

    if best_result is None:
        raise RuntimeError("No candidate k values were evaluated.")

    return best_result


def _golden_section_refine(
    *,
    samples: Sequence[Dict[str, float]],
    incumbent: FitResult,
    seed: Optional[FrictionParameters],
    seed_regularization: float,
    k_bounds: Tuple[float, float],
    model_coefficient: float,
    iterations: int = 40,
) -> FitResult:
    k_min, k_max = k_bounds
    left = max(k_min, incumbent.parameters.k / 1.8)
    right = min(k_max, incumbent.parameters.k * 1.8)
    if right - left < 1e-9:
        return incumbent

    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inverse_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0

    c = left + inverse_phi_sq * (right - left)
    d = left + inverse_phi * (right - left)
    c_result = _result_for_k(
        samples=samples,
        k_value=c,
        seed=seed,
        seed_regularization=seed_regularization,
        model_coefficient=model_coefficient,
    )
    d_result = _result_for_k(
        samples=samples,
        k_value=d,
        seed=seed,
        seed_regularization=seed_regularization,
        model_coefficient=model_coefficient,
    )
    best = min(
        [incumbent, c_result, d_result],
        key=lambda result: result.metrics["objective"],
    )

    for _ in range(iterations):
        if c_result.metrics["objective"] <= d_result.metrics["objective"]:
            right = d
            d = c
            d_result = c_result
            c = left + inverse_phi_sq * (right - left)
            c_result = _result_for_k(
                samples=samples,
                k_value=c,
                seed=seed,
                seed_regularization=seed_regularization,
                model_coefficient=model_coefficient,
            )
        else:
            left = c
            c = d
            c_result = d_result
            d = left + inverse_phi * (right - left)
            d_result = _result_for_k(
                samples=samples,
                k_value=d,
                seed=seed,
                seed_regularization=seed_regularization,
                model_coefficient=model_coefficient,
            )

        candidate_best = c_result if c_result.metrics["objective"] <= d_result.metrics["objective"] else d_result
        if candidate_best.metrics["objective"] < best.metrics["objective"]:
            best = candidate_best

    return best


def _fit_linear_terms_for_fixed_k(
    *,
    samples: Sequence[Dict[str, float]],
    fixed_k: float,
    seed: Optional[FrictionParameters],
    seed_regularization: float,
    model_coefficient: float,
) -> Tuple[FrictionParameters, Dict[str, float]]:
    normal_matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    normal_vector = [0.0, 0.0, 0.0]
    weighted_target_sum = 0.0
    total_weight = 0.0

    for sample in samples:
        velocity = sample["velocity"]
        target = sample["friction_torque"]
        weight = sample.get("weight", 1.0)
        basis = [
            math.tanh(model_coefficient * fixed_k * velocity),
            velocity,
            1.0,
        ]
        for row_index in range(3):
            normal_vector[row_index] += weight * basis[row_index] * target
            for col_index in range(3):
                normal_matrix[row_index][col_index] += weight * basis[row_index] * basis[col_index]
        weighted_target_sum += weight * target
        total_weight += weight

    if seed is not None and seed_regularization > 0.0:
        seed_vector = [seed.fc, seed.fv, seed.fo]
        seed_scales = [max(abs(seed.fc), 1.0), max(abs(seed.fv), 1.0), max(abs(seed.fo), 1.0)]
        for index in range(3):
            penalty = seed_regularization / (seed_scales[index] ** 2)
            normal_matrix[index][index] += penalty
            normal_vector[index] += penalty * seed_vector[index]

    solution = _solve_linear_system(normal_matrix, normal_vector)
    params = FrictionParameters(fc=solution[0], k=fixed_k, fv=solution[1], fo=solution[2])

    sse = 0.0
    sae = 0.0
    max_abs_error = 0.0
    weighted_mean = weighted_target_sum / max(total_weight, EPSILON)
    tss = 0.0
    for sample in samples:
        target = sample["friction_torque"]
        weight = sample.get("weight", 1.0)
        error = target - predict_friction(sample["velocity"], params, model_coefficient)
        sse += weight * error * error
        sae += weight * abs(error)
        max_abs_error = max(max_abs_error, abs(error))
        centered = target - weighted_mean
        tss += weight * centered * centered

    objective = sse
    if seed is not None and seed_regularization > 0.0:
        objective += seed_regularization * (
            ((params.fc - seed.fc) / max(abs(seed.fc), 1.0)) ** 2
            + ((params.fv - seed.fv) / max(abs(seed.fv), 1.0)) ** 2
            + ((params.fo - seed.fo) / max(abs(seed.fo), 1.0)) ** 2
            + ((params.k - seed.k) / max(abs(seed.k), 1.0)) ** 2
        )

    metrics = {
        "rmse": math.sqrt(sse / max(total_weight, EPSILON)),
        "mae": sae / max(total_weight, EPSILON),
        "max_abs_error": max_abs_error,
        "r2": 1.0 - sse / tss if tss > EPSILON else 1.0,
        "sse": sse,
        "objective": objective,
    }
    return params, metrics


def _result_for_k(
    *,
    samples: Sequence[Dict[str, float]],
    k_value: float,
    seed: Optional[FrictionParameters],
    seed_regularization: float,
    model_coefficient: float,
) -> FitResult:
    params, metrics = _fit_linear_terms_for_fixed_k(
        samples=samples,
        fixed_k=k_value,
        seed=seed,
        seed_regularization=seed_regularization,
        model_coefficient=model_coefficient,
    )
    return FitResult(parameters=params, metrics=metrics, sample_count=len(samples))


def _resolve_seed(
    seed_map: Dict[str, FrictionParameters],
    joint_name: str,
) -> Optional[FrictionParameters]:
    if joint_name in seed_map:
        return seed_map[joint_name]

    canonical = _canonical_joint_name(joint_name)
    if canonical in seed_map:
        return seed_map[canonical]

    return None


def _canonical_joint_name(joint_name: str) -> str:
    finger_match = re.search(r"(finger_joint\d+)$", joint_name)
    if finger_match:
        return finger_match.group(1)

    joint_match = re.search(r"(joint\d+)$", joint_name)
    if joint_match:
        return joint_match.group(1)

    return joint_name


def _pick(row: Dict[str, str], *names: str) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _parse_float(value: str, row_number: int, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number} has an invalid {field_name!r} value: {value!r}"
        ) from exc


def _parse_optional_float(
    value: Optional[str],
    row_number: int,
    field_name: str,
    *,
    default: float,
) -> float:
    if value is None:
        return default
    return _parse_float(value, row_number, field_name)


def _logspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1 or abs(stop - start) < EPSILON:
        return [start]

    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / (count - 1)
    return [math.exp(log_start + index * step) for index in range(count)]


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < EPSILON:
            augmented[pivot_index][pivot_index] += 1e-9
            pivot_row = max(
                range(pivot_index, size),
                key=lambda row: abs(augmented[row][pivot_index]),
            )
        if abs(augmented[pivot_row][pivot_index]) < EPSILON:
            raise ValueError("Unable to solve friction fit normal equations.")

        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if abs(factor) < EPSILON:
                continue
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[row][size] for row in range(size)]
