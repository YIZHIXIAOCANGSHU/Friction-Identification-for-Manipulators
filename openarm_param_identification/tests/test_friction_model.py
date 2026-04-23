import csv
import math
import os
import sys
import tempfile
import unittest


TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from friction_model import (  # noqa: E402
    FrictionParameters,
    fit_joint_model,
    load_samples_from_csv,
    predict_friction,
)


class FrictionModelTests(unittest.TestCase):
    def test_predict_friction_matches_openarm_formula(self):
        params = FrictionParameters(fc=0.4, k=25.0, fv=0.08, fo=-0.03)
        velocity = 1.2
        expected = 0.4 * math.tanh(0.1 * 25.0 * velocity) + 0.08 * velocity - 0.03
        self.assertAlmostEqual(predict_friction(velocity, params), expected, places=9)

    def test_fit_joint_model_recovers_parameters_from_synthetic_samples(self):
        ground_truth = FrictionParameters(fc=0.31, k=28.5, fv=0.065, fo=0.09)
        velocities = [x / 20.0 for x in range(-60, 61) if x != 0]
        samples = [
            {
                "velocity": velocity,
                "friction_torque": predict_friction(velocity, ground_truth),
                "weight": 1.0,
            }
            for velocity in velocities
        ]

        result = fit_joint_model(
            samples,
            seed=FrictionParameters(fc=0.306, k=28.417, fv=0.063, fo=0.088),
            seed_regularization=0.0,
        )

        self.assertAlmostEqual(result.parameters.fc, ground_truth.fc, delta=0.02)
        self.assertAlmostEqual(result.parameters.k, ground_truth.k, delta=2.0)
        self.assertAlmostEqual(result.parameters.fv, ground_truth.fv, delta=0.01)
        self.assertAlmostEqual(result.parameters.fo, ground_truth.fo, delta=0.01)
        self.assertLess(result.metrics["rmse"], 1e-6)

    def test_csv_loader_can_build_residual_friction_from_measured_torque(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "samples.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "joint",
                        "velocity",
                        "measured_torque",
                        "gravity_torque",
                        "coriolis_torque",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "joint": "joint1",
                        "velocity": "0.5",
                        "measured_torque": "1.40",
                        "gravity_torque": "1.00",
                        "coriolis_torque": "0.10",
                    }
                )

            loaded = load_samples_from_csv(
                csv_path,
                coriolis_scale=0.5,
            )

        self.assertIn("joint1", loaded)
        self.assertEqual(len(loaded["joint1"]), 1)
        self.assertAlmostEqual(loaded["joint1"][0]["friction_torque"], 0.35, places=9)


if __name__ == "__main__":
    unittest.main()
