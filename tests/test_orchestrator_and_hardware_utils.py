"""Validation tests for orchestrator, hardware diagnostics, env checker, and checkpoint helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tests._loader import load_script_module

env_mod = load_script_module("01_check_m5_env.py", "check_m5_env")
cam_mod = load_script_module("04_camera_stress_test.py", "camera_stress_test")
orch_mod = load_script_module("05_vlm_orchestrator_template.py", "vlm_orchestrator")
ckpt_mod = load_script_module("06_inspect_and_infer_checkpoint.py", "inspect_checkpoint")


class OrchestratorAndHardwareUtilsTests(unittest.TestCase):
    def test_orchestrator_recovers_on_empty_gripper_and_completes_on_grasp(self) -> None:
        orch = orch_mod.HackathonOrchestrator(checkpoints={})
        for step in range(450):
            # 1.3 = pinza cerrada vacia (RANGE_0_100), 13.8 = objeto sujetado
            gripper = 1.3 if step < 180 else 13.8
            tel = orch_mod.RobotTelemetry(
                joint_positions=np.zeros(6),
                gripper_pos=gripper,
                step_in_stage=orch.stage_step,
            )
            stage = orch.step_supervisor(tel)
            if stage == orch_mod.TaskStage.COMPLETED:
                break

        self.assertEqual(orch.stage, orch_mod.TaskStage.COMPLETED)
        self.assertEqual(orch.retries, 1)

    def test_orchestrator_aborts_safely_after_max_retries(self) -> None:
        orch = orch_mod.HackathonOrchestrator(checkpoints={}, max_retries=2)
        for _ in range(1000):
            tel = orch_mod.RobotTelemetry(
                joint_positions=np.zeros(6),
                gripper_pos=1.2,
                step_in_stage=orch.stage_step,
            )
            stage = orch.step_supervisor(tel)
            if stage == orch_mod.TaskStage.FAILED:
                break

        self.assertEqual(orch.stage, orch_mod.TaskStage.FAILED)
        self.assertEqual(orch.retries, 3)

    def test_serial_port_detection_supports_macos_and_linux(self) -> None:
        self.assertTrue(cam_mod.is_robot_serial_port("/dev/tty.usbmodem585A0076841"))
        self.assertTrue(cam_mod.is_robot_serial_port("/dev/tty.usbserial-0001"))
        self.assertTrue(cam_mod.is_robot_serial_port("/dev/ttyACM0"))
        self.assertTrue(cam_mod.is_robot_serial_port("/dev/ttyUSB0"))
        self.assertFalse(cam_mod.is_robot_serial_port("/dev/cu.debug-console"))
        self.assertFalse(cam_mod.is_robot_serial_port("/dev/cu.Bluetooth-Incoming-Port"))

    def test_camera_metrics_and_duplicate_index_rejection(self) -> None:
        self.assertEqual(cam_mod.run_camera_stress_test([0, 0], 640, 480, 30, 1), 1)

        # 31 timestamps a 30 Hz exactos (33.33 ms entre frames)
        ts = [i * (1.0 / 30.0) for i in range(31)]
        brightness = {0: [120.0, 123.5, 121.0]}
        healthy = cam_mod.compute_stream_metrics(ts, brightness, dropped_reads=0, target_fps=30)
        self.assertTrue(healthy["fps_ok"])
        self.assertTrue(healthy["jitter_ok"])
        self.assertAlmostEqual(healthy["real_fps"], 30.0, places=1)
        self.assertEqual(healthy["brightness_drift"][0], 3.5)

        # Caida a ~10 FPS con pico de 120 ms -> falla fps_ok y jitter_ok
        slow_ts = [0.0, 0.12, 0.24, 0.36, 0.48]
        degraded = cam_mod.compute_stream_metrics(slow_ts, {0: [100.0, 130.0]}, dropped_reads=4, target_fps=30)
        self.assertFalse(degraded["fps_ok"])
        self.assertFalse(degraded["jitter_ok"])

    def test_env_check_helpers_and_cli(self) -> None:
        self.assertTrue(env_mod.check_python_compatibility((3, 12, 5))[0])
        self.assertTrue(env_mod.check_python_compatibility((3, 13, 1))[0])
        self.assertFalse(env_mod.check_python_compatibility((3, 11, 9))[0])
        self.assertFalse(env_mod.check_python_compatibility((3, 14, 0))[0])

        self.assertFalse(env_mod.check_macos_mps_support("Darwin", "14.7")[0])
        self.assertTrue(env_mod.check_macos_mps_support("Darwin", "15.2")[0])
        self.assertTrue(env_mod.check_macos_mps_support("Darwin", "27.2.1")[0])

        self.assertGreaterEqual(env_mod.parse_semver_tuple("0.7.0"), (0, 7, 0))
        self.assertLess(env_mod.parse_semver_tuple("0.6.2"), (0, 7, 0))

        self.assertTrue(env_mod.check_disk_space_gb(".", min_gb=0.001)[0])
        self.assertFalse(env_mod.check_disk_space_gb(".", min_gb=1e9)[0])

        rc = env_mod.main(["--allow-cpu", "--skip-benchmark"])
        self.assertEqual(rc, 0)

    def test_checkpoint_extract_action_supports_tensor_and_policy_output(self) -> None:
        tensor = torch.zeros(1, 6)
        self.assertIs(ckpt_mod.extract_action_tensor(tensor), tensor)
        self.assertIs(ckpt_mod.extract_action_tensor({"action": tensor, "prediction": {}}), tensor)
        with self.assertRaises(KeyError):
            ckpt_mod.extract_action_tensor({"prediction": {}})


if __name__ == "__main__":
    unittest.main()
