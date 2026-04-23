# OpenArm Parameter Identification

This folder contains a standalone friction-parameter identification workflow built to match the
OpenArm teleoperation friction model:

`tau_f = Fc * tanh(0.1 * k * dq) + Fv * dq + Fo`

It is intentionally independent from the upstream repositories in this directory so you can iterate
on identification logic without modifying the downloaded OpenArm sources.

## What it does

- Reuses the same friction model used by `openarm_teleop`
- Accepts existing OpenArm parameters as seeds
- Fits improved `Fc`, `k`, `Fv`, and `Fo` values from offline CSV data
- Can output per-sample residuals for debugging and data cleaning
- Uses only the Python standard library

## Files

- `friction_model.py`
  Core fitting logic and CSV loading helpers
- `fit_friction_model.py`
  Command-line tool for joint-wise parameter fitting
- `seeds/follower_seed.json`
  Seed parameters copied from `openarm_teleop/config/follower.yaml`
- `seeds/leader_seed.json`
  Seed parameters copied from `openarm_teleop/config/leader.yaml`
- `samples/friction_samples_template.csv`
  Minimal CSV template for new datasets
- `tests/test_friction_model.py`
  Regression tests for the fitter

## Recommended data format

Preferred columns:

```csv
joint,velocity,friction_torque,weight
joint1,-1.2,-0.29,1.0
joint1,1.2,0.41,1.0
```

You can also provide measured torque and let the script build a residual target:

```csv
joint,velocity,measured_torque,gravity_torque,coriolis_torque,weight
joint1,0.5,1.40,1.00,0.10,1.0
```

In that case the script computes:

`friction_torque = measured_torque - gravity_torque - coriolis_scale * coriolis_torque`

## Usage

Fit from a CSV using the current follower parameters as seeds:

```bash
cd /home/luwen/桌面/openarm的参数/openarm_param_identification
python3 fit_friction_model.py \
  --csv your_samples.csv \
  --seed-json seeds/follower_seed.json \
  --output fitted_friction_params.json \
  --predictions-csv fitted_predictions.csv
```

Fit only one joint:

```bash
python3 fit_friction_model.py \
  --csv your_samples.csv \
  --seed-json seeds/follower_seed.json \
  --joint joint3
```

Disable seed regularization if you want a freer fit:

```bash
python3 fit_friction_model.py \
  --csv your_samples.csv \
  --seed-json seeds/follower_seed.json \
  --seed-regularization 0.0
```

## Output

The output JSON contains:

- fitted parameters for each joint
- RMSE, MAE, max absolute error, SSE, and R2
- sample count

## Data-collection advice

- Record both positive and negative joint velocities, otherwise `Fc` and `Fo` become harder to
  separate.
- Include low-speed and high-speed segments, otherwise `k` and `Fv` become weakly observable.
- If possible, log residual torque after removing gravity and known feedforward terms.
- Filter out saturation, collision, and strong external-contact intervals before fitting.

## Verification

Run:

```bash
python3 -m unittest discover -s tests
```
