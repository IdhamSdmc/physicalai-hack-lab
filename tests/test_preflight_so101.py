"""Validation tests for `scripts/09_preflight_so101.py` (env, calibration, CLI contracts)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests._loader import REPO_ROOT, load_script_module

preflight = load_script_module("09_preflight_so101.py", "preflight_so101")


def _valid_env() -> dict[str, str]:
    return {
        "ROBOT_TYPE": "so101_follower",
        "TELEOP_TYPE": "so101_leader",
        "ROBOT_ID": "bilbao_follower",
        "TELEOP_ID": "bilbao_leader",
        "FOLLOWER_PORT": "/dev/tty.usbmodem101",
        "LEADER_PORT": "/dev/tty.usbmodem102",
        "ROBOT_CAMERAS": "{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }",
        "DATASET_REPO": "team/bilbao_task1",
        "ROLLOUT_DATASET_REPO": "team/rollout_bilbao_task1",
        "DAGGER_DATASET_REPO": "team/rollout_bilbao_task1_dagger",
        "TASK_DESC": "pick red cube",
        "HACK_DEVICE": "cpu",
    }


def _valid_calibration(offset: int = 0) -> dict[str, dict[str, int]]:
    return {
        name: {
            "id": idx,
            "drive_mode": 0,
            "homing_offset": 10 + offset + idx,
            "range_min": 0 if name in preflight.FULL_TURN_MOTORS else 500 + offset,
            "range_max": 4095 if name in preflight.FULL_TURN_MOTORS else 3500 + offset,
        }
        for idx, name in enumerate(preflight.SO101_MOTORS, start=1)
    }


class PreflightSO101Tests(unittest.TestCase):
    def test_valid_env_and_lerobot_cli_contracts_pass(self) -> None:
        env = _valid_env()
        res = preflight.validate_hackathon_env(env)
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["warnings"], [])
        cli_errors = preflight.validate_lerobot_cli_contracts(env)
        self.assertEqual(cli_errors, [])

    def test_detects_critical_env_misconfigurations(self) -> None:
        # 1. IDs iguales (el fallo silencioso que advierte la guia SO-101)
        env = _valid_env()
        env["TELEOP_ID"] = env["ROBOT_ID"]
        res = preflight.validate_hackathon_env(env)
        self.assertTrue(any("ROBOT_ID == TELEOP_ID" in err for err in res["errors"]))

        # 2. Puertos iguales
        env = _valid_env()
        env["LEADER_PORT"] = env["FOLLOWER_PORT"]
        res = preflight.validate_hackathon_env(env)
        self.assertTrue(any("mismo puerto" in err for err in res["errors"]))

        # 3. Indice de camara duplicado
        env = _valid_env()
        env["ROBOT_CAMERAS"] = "{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }"
        res = preflight.validate_hackathon_env(env)
        self.assertTrue(any("duplicado" in err for err in res["errors"]))

        # 4. Resolucion alta -> aviso USB 2.0
        env = _valid_env()
        env["ROBOT_CAMERAS"] = "{ top: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30} }"
        res = preflight.validate_hackathon_env(env)
        self.assertEqual(res["errors"], [])
        self.assertTrue(any("USB 2.0" in w for w in res["warnings"]))

        # 5. Falta prefijo rollout_ en rollout/dagger o hay prefijo eval_ en record
        env = _valid_env()
        env["DATASET_REPO"] = "team/eval_task"
        env["ROLLOUT_DATASET_REPO"] = "team/bad_rollout"
        env["DAGGER_DATASET_REPO"] = "team/bad_dagger"
        res = preflight.validate_hackathon_env(env)
        self.assertEqual(len(res["errors"]), 3)

    def test_calibration_pair_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = Path(tmp)
            follower_dir = cal_dir / "robots" / "so_follower"
            leader_dir = cal_dir / "teleoperators" / "so_leader"
            follower_dir.mkdir(parents=True)
            leader_dir.mkdir(parents=True)

            # Sin ficheros: warning con require_files=False, error con require_files=True
            res_opt = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=False)
            self.assertEqual(res_opt["errors"], [])
            self.assertEqual(len(res_opt["warnings"]), 2)

            res_req = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=True)
            self.assertEqual(len(res_req["errors"]), 2)

            # Calibraciones validas y distintas
            f_path = follower_dir / "f1.json"
            l_path = leader_dir / "l1.json"
            f_path.write_text(json.dumps(_valid_calibration(offset=0)))
            l_path.write_text(json.dumps(_valid_calibration(offset=100)))
            res_ok = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=True)
            self.assertEqual(res_ok["errors"], [])
            self.assertEqual(res_ok["warnings"], [])

            # Calibracion identica -> error de copia/intercambio
            l_path.write_text(json.dumps(_valid_calibration(offset=0)))
            res_dup = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=True)
            self.assertTrue(any("idéntica" in err for err in res_dup["errors"]))

            # Articulacion faltante + rango invertido + homing_offset fuera de rango STS3215
            broken = _valid_calibration(offset=50)
            del broken["gripper"]
            broken["shoulder_pan"]["range_min"] = 3900
            broken["shoulder_lift"]["homing_offset"] = 3000
            l_path.write_text(json.dumps(broken))
            res_bad = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=True)
            self.assertTrue(any("gripper" in err for err in res_bad["errors"]))
            self.assertTrue(any("range_min" in err for err in res_bad["errors"]))
            self.assertTrue(any("homing_offset" in err for err in res_bad["errors"]))

            # Calibracion fisicamente incompleta (recorrido corto, homing descentrado, wrist_roll no full-turn, divergencia entre brazos)
            sloppy = _valid_calibration(offset=50)
            sloppy["elbow_flex"]["range_min"] = 2000
            sloppy["elbow_flex"]["range_max"] = 2060  # span=60 (<100) y divergencia >1000 vs follower
            sloppy["shoulder_pan"]["range_min"] = 2200  # homing 2047 fuera de [2200, 3550]
            sloppy["wrist_roll"]["range_min"] = 200
            l_path.write_text(json.dumps(sloppy))
            res_warn = preflight.validate_calibration_pair(cal_dir, "f1", "l1", require_files=True)
            self.assertEqual(res_warn["errors"], [])
            self.assertTrue(any("recorrido muy corto" in w for w in res_warn["warnings"]))
            self.assertTrue(any("centro homing" in w for w in res_warn["warnings"]))
            self.assertTrue(any("full_turn" in w for w in res_warn["warnings"]))
            self.assertTrue(any("Diferencia grande de recorrido" in w for w in res_warn["warnings"]))

    def test_cli_exit_codes_for_valid_and_invalid_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = Path(tmp)
            env = os.environ.copy()
            env.update(_valid_env())

            ok_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "09_preflight_so101.py"),
                    "--use-current-env",
                    "--skip-cli-parse",
                    "--calibration-dir",
                    str(cal_dir),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ok_proc.returncode, 0, ok_proc.stdout + ok_proc.stderr)

            bad_env = env.copy()
            bad_env["TELEOP_ID"] = bad_env["ROBOT_ID"]
            bad_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "09_preflight_so101.py"),
                    "--use-current-env",
                    "--skip-cli-parse",
                    "--calibration-dir",
                    str(cal_dir),
                ],
                env=bad_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(bad_proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
