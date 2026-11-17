"""Validation tests for Hackathon documentation, shell scripts, and LeRobot CLI contracts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock

from tests._loader import REPO_ROOT, load_script_module

preflight = load_script_module("09_preflight_so101.py", "preflight_so101")


def _extract_bash_commands(markdown_text: str) -> list[str]:
    """Extract joined bash commands (resolving backslash line continuations) from markdown fences."""
    commands: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", markdown_text, flags=re.DOTALL):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
    return commands


def _expand_env(token: str, env: dict[str, str]) -> str:
    def _repl(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        return env.get(key, match.group(0))

    return re.sub(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)", _repl, token)


class HackathonDocsAndCLIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = preflight.load_env_from_script(REPO_ROOT / "scripts" / "env_hackathon.sh")

    def test_env_hackathon_satisfies_preflight_invariants(self) -> None:
        result = preflight.validate_hackathon_env(self.env)
        self.assertEqual(result["errors"], [], f"Unexpected env errors: {result['errors']}")
        self.assertEqual(result["warnings"], [], f"Unexpected env warnings: {result['warnings']}")
        self.assertIn("top", result["parsed_cameras"])
        self.assertIn("wrist", result["parsed_cameras"])

    def test_shell_scripts_pass_bash_syntax_check(self) -> None:
        for script_name in ("00_setup_env.sh", "03_train_local_m5.sh", "env_hackathon.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(REPO_ROOT / "scripts" / script_name)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"bash -n failed for {script_name}: {proc.stderr}")

    def test_train_script_emits_expected_flags_for_all_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "lerobot-train"
            fake_bin.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
            fake_bin.chmod(0o755)

            run_env = os.environ.copy()
            run_env["LEROBOT_TRAIN"] = str(fake_bin)

            smoke = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "03_train_local_m5.sh"), "smoke", "25", "team/demo"],
                env=run_env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--dataset.repo_id=team/demo", smoke.stdout)
            self.assertIn("--policy.device=mps", smoke.stdout)
            self.assertIn("--trackio.enable=false", smoke.stdout)
            self.assertIn("--steps=25", smoke.stdout)

            so101 = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "03_train_local_m5.sh"), "so101", "50", "team/demo"],
                env=run_env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--dataset.image_transforms.enable=true", so101.stdout)
            self.assertIn("--num_workers=2", so101.stdout)

            train_run = tmp_path / "run1"
            cfg_file = train_run / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            cfg_file.parent.mkdir(parents=True)
            cfg_file.write_text("{}")

            resumed = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "03_train_local_m5.sh"), "resume", str(train_run)],
                env=run_env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn(f"--config_path={cfg_file}", resumed.stdout)
            self.assertIn("--resume=true", resumed.stdout)

            bad = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "03_train_local_m5.sh"), "invalid_mode"],
                env=run_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_cheatsheet_commands_comply_with_lerobot_v070_rules(self) -> None:
        cheatsheet_text = (REPO_ROOT / "CHEATSHEET_BILBAO_48H.md").read_text()
        commands = _extract_bash_commands(cheatsheet_text)
        lerobot_cmds = [cmd for cmd in commands if cmd.startswith("lerobot-")]
        self.assertGreaterEqual(len(lerobot_cmds), 8)

        for cmd in lerobot_cmds:
            self.assertNotIn("--wandb.", cmd, f"Deprecated --wandb flag found in: {cmd}")
            self.assertNotIn("--eval_freq", cmd, f"Deprecated --eval_freq flag found in: {cmd}")
            tokens = [_expand_env(t, self.env) for t in shlex.split(cmd)]
            exe = tokens[0]

            if exe == "lerobot-record":
                self.assertFalse(
                    any(t.startswith("--policy.path") for t in tokens),
                    f"lerobot-record must not take --policy.path: {cmd}",
                )
                for tok in tokens:
                    if tok.startswith("--dataset.repo_id="):
                        repo = tok.split("=", 1)[1]
                        leaf = repo.split("/", 1)[-1]
                        self.assertFalse(leaf.startswith("eval_"), f"eval_ forbidden in lerobot-record: {cmd}")

            if exe == "lerobot-rollout":
                for tok in tokens:
                    if tok.startswith("--dataset.repo_id="):
                        repo = tok.split("=", 1)[1]
                        leaf = repo.split("/", 1)[-1]
                        self.assertTrue(
                            leaf.startswith("rollout_"),
                            f"lerobot-rollout dataset must start with 'rollout_': {cmd}",
                        )

    def test_lerobot_dataset_stamping_and_rollout_prefix_contracts(self) -> None:
        from lerobot.configs.dataset import DatasetRecordConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.rollout.configs import RolloutConfig
        from lerobot.rollout.context import _build_rollout_dataset
        from lerobot.scripts.lerobot_record import RecordConfig

        stamped = DatasetRecordConfig(repo_id="team/demo", no_stamp=False)
        stamped.stamp_repo_id()
        self.assertRegex(stamped.repo_id, r"^team/demo_\d{8}_\d{6}$")

        exact = DatasetRecordConfig(repo_id="team/demo", no_stamp=True)
        exact.stamp_repo_id()
        self.assertEqual(exact.repo_id, "team/demo")

        with self.assertRaises(ValueError):
            RecordConfig(
                robot=MagicMock(),
                dataset=DatasetRecordConfig(repo_id="team/demo"),
                teleop=None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp)
            (policy_dir / "config.json").write_text('{"type": "act", "chunk_size": 50, "n_action_steps": 50, "device": "cpu"}')
            bad_rollout_cfg = preflight.parse_lerobot_config(
                RolloutConfig,
                [
                    "--strategy.type=dagger",
                    f"--policy.path={policy_dir}",
                    "--device=cpu",
                    "--robot.type=so101_follower",
                    "--robot.port=/dev/ttyACM0",
                    "--teleop.type=so101_leader",
                    "--teleop.port=/dev/ttyACM1",
                    "--dataset.repo_id=team/bilbao_dagger",
                    "--dataset.single_task=pick",
                ],
            )
            with self.assertRaisesRegex(ValueError, "rollout_"):
                _build_rollout_dataset(bad_rollout_cfg, MagicMock(), {})

        act_ok = ACTConfig(chunk_size=50, n_action_steps=1, temporal_ensemble_coeff=0.01)
        self.assertEqual(act_ok.temporal_ensemble_coeff, 0.01)
        with self.assertRaises(NotImplementedError):
            ACTConfig(chunk_size=50, n_action_steps=50, temporal_ensemble_coeff=0.01)


if __name__ == "__main__":
    unittest.main()
