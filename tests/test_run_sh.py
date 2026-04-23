from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SH_PATH = PROJECT_ROOT / "run.sh"


class RunShInteractionTests(unittest.TestCase):
    def _make_env_with_fake_python(self) -> tuple[dict[str, str], Path]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        fake_dir = Path(tempdir.name)
        fake_python = fake_dir / "python3"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{fake_dir}:{env.get('PATH', '')}"
        return env, fake_dir

    def _run_script(
        self,
        *args: str,
        input_text: str = "",
    ) -> subprocess.CompletedProcess[str]:
        env, _ = self._make_env_with_fake_python()
        return subprocess.run(
            [str(RUN_SH_PATH), *args],
            input=input_text,
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )

    def _fake_python_args(self, completed: subprocess.CompletedProcess[str]) -> list[str]:
        lines = completed.stdout.strip().splitlines()
        try:
            start = len(lines) - 1 - lines[::-1].index("-m")
        except ValueError:
            return []
        return lines[start:]

    def test_help_prints_usage(self) -> None:
        completed = self._run_script("--help")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)
        self.assertIn("./run.sh", completed.stdout)

    def test_interactive_menu_builds_expected_command(self) -> None:
        completed = self._run_script(
            input_text="\n".join(
                [
                    "1",
                    "2",
                    "1,3,4",
                    "",
                ]
            ),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("最终命令", completed.stdout)
        self.assertEqual(
            self._fake_python_args(completed),
            [
                "-m",
                "friction_identification_core",
                "--mode",
                "identify",
                "--config",
                "friction_identification_core/default.yaml",
                "--motors",
                "1,3,4",
                "--output",
                "results",
            ],
        )
        self.assertIn("--mode", completed.stdout)
        self.assertIn("--motors", completed.stdout)
        self.assertIn("--output", completed.stdout)

    def test_compensate_menu_builds_expected_command(self) -> None:
        completed = self._run_script(
            input_text="\n".join(
                [
                    "2",
                    "1",
                    "",
                ]
            ),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(
            self._fake_python_args(completed),
            [
                "-m",
                "friction_identification_core",
                "--mode",
                "compensate",
                "--config",
                "friction_identification_core/default.yaml",
                "--motors",
                "all",
                "--output",
                "results",
            ],
        )

    def test_invalid_inputs_retry_until_valid(self) -> None:
        completed = self._run_script(
            input_text="\n".join(
                [
                    "",
                    "9",
                    "1",
                    "9",
                    "2",
                    "1, x",
                    "all",
                    "",
                ]
            ),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("输入无效", completed.stdout)
        self.assertIn("--config", completed.stdout)
        self.assertIn("friction_identification_core/default.yaml", completed.stdout)
        self.assertIn("--motors", completed.stdout)
        self.assertEqual(
            self._fake_python_args(completed),
            [
                "-m",
                "friction_identification_core",
                "--mode",
                "identify",
                "--config",
                "friction_identification_core/default.yaml",
                "--motors",
                "all",
                "--output",
                "results",
            ],
        )

    def test_mode_zero_exits_without_running_python(self) -> None:
        completed = self._run_script(input_text="0\n")

        self.assertEqual(completed.returncode, 0)
        self.assertIn("已退出", completed.stdout)
        self.assertEqual(self._fake_python_args(completed), [])

    def test_legacy_non_interactive_invocation_is_rejected(self) -> None:
        completed = self._run_script("sequential")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("python3 -m friction_identification_core", completed.stderr)


if __name__ == "__main__":
    unittest.main()
