"""Unit tests for the OpenCV triage vision calibrator and Clean-First supervisor."""

from __future__ import annotations

from pathlib import Path
import os
import signal
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from tests._loader import load_script_module

supervisor = load_script_module("10_opencv_triage_supervisor.py", "opencv_triage_supervisor")


def _make_synthetic_triage_frame(
    config: dict,
    green_boxes: list[tuple[int, int, int, int]] | None = None,
    red_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Create a 640x480 BGR synthetic table image with vivid green and red cubes."""
    frame = np.full((config["height"], config["width"], 3), (220, 220, 220), dtype=np.uint8)
    for x, y, w, h in green_boxes or []:
        # Vivid green in BGR (H ~ 60 in OpenCV HSV)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 190, 20), -1)
    for x, y, w, h in red_boxes or []:
        # Vivid red in BGR (H ~ 0/180 in OpenCV HSV)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 210), -1)
    return frame


class OpenCVTriageSupervisorTests(unittest.TestCase):
    def test_detects_green_and_red_cubes_across_rois_and_filters_noise(self) -> None:
        cfg = supervisor.default_vision_config(cam_top_idx=1)
        # Put 1 green + 1 red inside ZONA_A, 1 green in ZONA_B_SUP, 1 red in ZONA_B_INF, plus 4x4 noise dot
        frame = _make_synthetic_triage_frame(
            cfg,
            green_boxes=[
                (350, 120, 34, 34),  # ZONA_A
                (180, 190, 34, 34),  # ZONA_B_SUP
                (360, 220, 4, 4),    # Tiny noise (< min_area_px)
            ],
            red_boxes=[
                (410, 170, 34, 34),  # ZONA_A
                (190, 70, 34, 34),   # ZONA_B_INF
            ],
        )

        analysis = supervisor.analyze_frame(frame, cfg)
        self.assertEqual(analysis["verdes_en_A"], 1)
        self.assertEqual(analysis["rojos_en_A"], 1)
        self.assertEqual(analysis["verdes_en_B_SUP"], 1)
        self.assertEqual(analysis["rojos_en_B_INF"], 1)
        self.assertEqual(analysis["target"], "sano")
        self.assertEqual(analysis["next_order"], "EVACUAR SANO -> act_sano")

        hud = supervisor.render_hud(frame, cfg, analysis, active_roi="ZONA_A")
        self.assertEqual(hud.shape, frame.shape)

    def test_clean_first_priority_switches_to_contaminado_then_completed(self) -> None:
        self.assertEqual(supervisor.determine_next_action(2, 2), ("sano", "EVACUAR SANO -> act_sano"))
        self.assertEqual(supervisor.determine_next_action(1, 2), ("sano", "EVACUAR SANO -> act_sano"))
        self.assertEqual(
            supervisor.determine_next_action(0, 2),
            ("contaminado", "CUARENTENA + VASO D -> act_contaminado"),
        )
        self.assertEqual(
            supervisor.determine_next_action(0, 0),
            ("completed", "ZONA A LIMPIA - COMPLETADO"),
        )

    def test_save_and_load_vision_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "triage_vision_config.json"
            cfg = supervisor.default_vision_config(cam_top_idx=1)
            cfg["rois"]["ZONA_A"] = [300, 80, 500, 290]
            cfg["min_area_px"] = 250
            supervisor.save_vision_config(cfg, cfg_path)

            reloaded = supervisor.load_vision_config(cfg_path)
            self.assertEqual(reloaded["rois"]["ZONA_A"], [300, 80, 500, 290])
            self.assertEqual(reloaded["min_area_px"], 250)
            self.assertEqual(reloaded["camera_top_index"], 1)

    def test_build_rollout_command_uses_dataset_camera_keys_and_safety_flags(self) -> None:
        cmd = supervisor.build_rollout_command(
            policy_path=Path("outputs/train/act_sano/checkpoints/last/pretrained_model"),
            duration_s=15.0,
            robot_port="/dev/tty.usbmodem5B610343371",
            robot_id="bilbao_follower",
            robot_type="so101_follower",
            cam_top_idx=1,
            cam_wrist_idx=0,
        )
        joined = " ".join(cmd)
        self.assertIn("camera_top", joined)
        self.assertIn("camera_robot", joined)
        self.assertIn("--device=mps", joined)
        self.assertIn("--robot.max_relative_target=30.0", joined)
        self.assertIn("--robot.disable_torque_on_disconnect=false", joined)
        self.assertIn("--policy.n_action_steps=50", joined)
        self.assertIn("--display_data=false", joined)
        self.assertIn("--return_to_initial_position=true", joined)

    def test_autonomous_supervisor_retries_on_failed_evacuation_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p_sano = tmp_path / "act_sano"
            p_cont = tmp_path / "act_contaminado"
            p_sano.mkdir()
            p_cont.mkdir()

            # Sequence of scans:
            # 1) Initial: 1 green, 1 red -> runs sano
            # 2) Post-sano attempt 1: still 1 green (dropped/missed!) -> triggers AUTO-RECOVERY
            # 3) Pre-retry scan: still 1 green, 1 red -> runs sano retry #1
            # 4) Post-sano retry: 0 green, 1 red -> sano verified!
            # 5) Next cycle: 0 green, 1 red -> runs contaminado
            # 6) Post-contaminado: 0 green, 0 red -> verified!
            # 7) Next cycle: 0 green, 0 red -> completed!
            states = iter(
                [
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (0, 1),
                    (0, 1),
                    (0, 0),
                    (0, 0),
                ]
            )

            def fake_scan() -> dict:
                v, r = next(states)
                target, order = supervisor.determine_next_action(v, r)
                return {
                    "verdes_en_A": v,
                    "rojos_en_A": r,
                    "verdes_en_B_SUP": 0,
                    "rojos_en_B_INF": 0,
                    "target": target,
                    "next_order": order,
                    "cubes": [],
                }

            executed_policies: list[str] = []

            def fake_runner(cmd: list[str]) -> int:
                for arg in cmd:
                    if arg.startswith("--policy.path="):
                        executed_policies.append(Path(arg.split("=", 1)[1]).name)
                return 0

            rc = supervisor.run_autonomous_supervisor(
                config=supervisor.default_vision_config(cam_top_idx=1),
                policy_sano=p_sano,
                policy_contaminado=p_cont,
                robot_port="/dev/tty.usbmodem5B610343371",
                robot_id="bilbao_follower",
                robot_type="so101_follower",
                cam_wrist_idx=0,
                max_retries=2,
                scanner_fn=fake_scan,
                runner_fn=fake_runner,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(executed_policies, ["act_sano", "act_sano", "act_contaminado"])

    def test_detect_scene_change_flags_shifted_coordinates_without_count_change(self) -> None:
        cfg = supervisor.default_vision_config(cam_top_idx=1)
        frame1 = _make_synthetic_triage_frame(cfg, green_boxes=[(350, 120, 34, 34)], red_boxes=[(410, 170, 34, 34)])
        frame2 = _make_synthetic_triage_frame(cfg, green_boxes=[(385, 145, 34, 34)], red_boxes=[(410, 170, 34, 34)])
        map1 = supervisor.build_scene_map(supervisor.analyze_frame(frame1, cfg))
        map2 = supervisor.build_scene_map(supervisor.analyze_frame(frame2, cfg))

        change = supervisor.detect_scene_change(map1, map2, move_threshold_px=12.0)
        self.assertTrue(change["changed"])
        self.assertFalse(change["counts_changed"])
        self.assertEqual(len(change["moved_objects"]), 1)
        self.assertEqual(change["moved_objects"][0]["color"], "S")
        self.assertGreater(change["max_shift_px"], 40.0)

    def test_supervisor_remaps_and_recovers_when_bad_grasp_shifts_cube_coordinates(self) -> None:
        cfg = supervisor.default_vision_config(cam_top_idx=1)
        frame_init = _make_synthetic_triage_frame(cfg, green_boxes=[(350, 120, 34, 34)], red_boxes=[])
        frame_nudged = _make_synthetic_triage_frame(cfg, green_boxes=[(385, 145, 34, 34)], red_boxes=[])
        frame_empty = _make_synthetic_triage_frame(cfg, green_boxes=[], red_boxes=[])

        scans = iter(
            [
                supervisor.analyze_frame(frame_init, cfg),
                supervisor.analyze_frame(frame_nudged, cfg),
                supervisor.analyze_frame(frame_nudged, cfg),
                supervisor.analyze_frame(frame_empty, cfg),
                supervisor.analyze_frame(frame_empty, cfg),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p_sano = tmp_path / "act_sano"
            p_cont = tmp_path / "act_contaminado"
            p_sano.mkdir()
            p_cont.mkdir()

            executed: list[str] = []

            def fake_runner(cmd: list[str]) -> int:
                executed.append(Path([a.split("=", 1)[1] for a in cmd if a.startswith("--policy.path=")][0]).name)
                return 0

            rc = supervisor.run_autonomous_supervisor(
                config=cfg,
                policy_sano=p_sano,
                policy_contaminado=p_cont,
                robot_port="/dev/tty.usbmodem5B610343371",
                robot_id="bilbao_follower",
                robot_type="so101_follower",
                cam_wrist_idx=0,
                remap_on_change=True,
                scanner_fn=lambda: next(scans),
                runner_fn=fake_runner,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(executed, ["act_sano", "act_sano"])

    def test_sigterm_stops_supervisor_cleanly_and_restores_signal_handler(self) -> None:
        cfg = supervisor.default_vision_config(cam_top_idx=1)
        frame = _make_synthetic_triage_frame(cfg, green_boxes=[(350, 120, 34, 34)], red_boxes=[])
        prev_handler = signal.getsignal(signal.SIGTERM)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p_sano = tmp_path / "act_sano"
            p_cont = tmp_path / "act_contaminado"
            p_sano.mkdir()
            p_cont.mkdir()

            def trigger_sigterm(_cmd: list[str]) -> int:
                os.kill(os.getpid(), signal.SIGTERM)
                return 0

            rc = supervisor.run_autonomous_supervisor(
                config=cfg,
                policy_sano=p_sano,
                policy_contaminado=p_cont,
                robot_port="/dev/tty.usbmodem5B610343371",
                robot_id="bilbao_follower",
                robot_type="so101_follower",
                cam_wrist_idx=0,
                scanner_fn=lambda: supervisor.analyze_frame(frame, cfg),
                runner_fn=trigger_sigterm,
            )
            self.assertEqual(rc, 128 + signal.SIGTERM)
            self.assertIs(signal.getsignal(signal.SIGTERM), prev_handler)

    def test_persistent_flag_default_and_missing_checkpoint_guard(self) -> None:
        args_default = supervisor.parse_args(["--mode", "run"])
        self.assertFalse(args_default.subprocess)
        args_sub = supervisor.parse_args(["--mode", "run", "--subprocess"])
        self.assertTrue(args_sub.subprocess)

        with tempfile.TemporaryDirectory() as tmp:
            missing_p = Path(tmp) / "missing_model"
            rc = supervisor.run_autonomous_supervisor(
                config=supervisor.default_vision_config(cam_top_idx=1),
                policy_sano=missing_p,
                policy_contaminado=missing_p,
                robot_port="/dev/tty.usbmodem5B610343371",
                robot_id="bilbao_follower",
                robot_type="so101_follower",
                cam_wrist_idx=0,
                persistent=True,
                dry_run=False,
            )
            self.assertEqual(rc, 2)

    def test_persistent_session_warmup_rebuilds_frame_per_model(self) -> None:
        class _DummyPolicy:
            def __init__(self) -> None:
                self.config = type("Cfg", (), {"temporal_ensemble_coeff": None, "n_action_steps": 50})()
                self.select_calls = 0

            def to(self, _device):
                return self

            def eval(self):
                return self

            def reset(self):
                return None

            def select_action(self, batch):
                import torch

                self.select_calls += 1
                self.last_batch_shape = tuple(batch["observation.state"].shape)
                return torch.zeros((1, 6), dtype=torch.float32)

        class _DummyProc:
            def __call__(self, x):
                return x

            def reset(self):
                return None

        class _DummyRobot:
            def __init__(self, _cfg) -> None:
                self.observation_features = {
                    "shoulder_pan.pos": float,
                    "shoulder_lift.pos": float,
                    "elbow_flex.pos": float,
                    "wrist_flex.pos": float,
                    "wrist_roll.pos": float,
                    "gripper.pos": float,
                    "camera_top": (48, 64, 3),
                    "camera_robot": (48, 64, 3),
                }
                self.action_features = {k: float for k in self.observation_features if k.endswith(".pos")}

            def connect(self, calibrate: bool = False) -> None:
                return None

            def get_observation(self) -> dict:
                return {
                    "shoulder_pan.pos": 0.0,
                    "shoulder_lift.pos": -50.0,
                    "elbow_flex.pos": 80.0,
                    "wrist_flex.pos": 40.0,
                    "wrist_roll.pos": 0.0,
                    "gripper.pos": 30.0,
                    "camera_top": np.zeros((48, 64, 3), dtype=np.uint8),
                    "camera_robot": np.zeros((48, 64, 3), dtype=np.uint8),
                }

            def disconnect(self) -> None:
                return None

        policies = [_DummyPolicy(), _DummyPolicy()]
        policy_iter = iter(policies)

        session = supervisor.PersistentTriageSession(
            policy_sano_path=Path("dummy_sano"),
            policy_contaminado_path=Path("dummy_cont"),
            robot_port="/dev/tty.dummy",
            robot_id="bilbao_follower",
            robot_type="so101_follower",
            cam_top_idx=1,
            cam_wrist_idx=0,
            width=64,
            height=48,
            device="cpu",
        )
        with (
            mock.patch("lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained", side_effect=lambda _p: next(policy_iter)),
            mock.patch("lerobot.policies.factory.make_pre_post_processors", side_effect=lambda *_a, **_kw: (_DummyProc(), _DummyProc())),
            mock.patch("lerobot.robots.so_follower.SOFollower", side_effect=_DummyRobot),
        ):
            session.start()
        try:
            self.assertEqual(policies[0].select_calls, 1)
            self.assertEqual(policies[1].select_calls, 1)
            self.assertEqual(policies[0].last_batch_shape, (1, 6))
            self.assertEqual(policies[1].last_batch_shape, (1, 6))
        finally:
            session.close()

    def test_early_stop_tracker_guards_occlusion_and_requires_vaso_d_for_contaminado(self) -> None:
        sano_tracker = supervisor.EarlyStopTracker(
            "sano",
            {"verdes_en_A": 2, "rojos_en_A": 2, "verdes_en_B_SUP": 0, "rojos_en_B_INF": 0},
        )
        # Too early (< 7s)
        self.assertFalse(
            sano_tracker.update_vision(
                {"verdes_en_A": 1, "rojos_en_A": 2, "verdes_en_B_SUP": 1, "rojos_en_B_INF": 0},
                {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -40.0},
                elapsed_s=4.0,
            )
        )
        # Arm occludes ZONA_A without cube appearing in B_SUP -> must NOT stop
        for _ in range(4):
            self.assertFalse(
                sano_tracker.update_vision(
                    {"verdes_en_A": 1, "rojos_en_A": 2, "verdes_en_B_SUP": 0, "rojos_en_B_INF": 0},
                    {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -40.0},
                    elapsed_s=9.0,
                )
            )
        # Confirmed in B_SUP + arm clear requires 3 consecutive checks
        an_ok = {"verdes_en_A": 1, "rojos_en_A": 2, "verdes_en_B_SUP": 1, "rojos_en_B_INF": 0}
        self.assertFalse(sano_tracker.update_vision(an_ok, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -40.0}, 10.0))
        self.assertFalse(sano_tracker.update_vision(an_ok, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -40.0}, 10.3))
        self.assertTrue(sano_tracker.update_vision(an_ok, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -40.0}, 10.6))
        self.assertIn("Almacén", sano_tracker.stop_reason or "")

        cont_tracker = supervisor.EarlyStopTracker(
            "contaminado",
            {"verdes_en_A": 0, "rojos_en_A": 2, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 0},
        )
        # Vaso D dip BEFORE cube is confirmed in Desechos must NOT count
        self.assertFalse(cont_tracker.observe_robot_state({"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 25.0}, 8.0))
        self.assertFalse(cont_tracker.visited_vaso_d)
        # Cube lands in Desechos (2 checks) -> dropped_in_desechos True, still NO early stop yet
        an_c = {"verdes_en_A": 0, "rojos_en_A": 1, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 1}
        self.assertFalse(cont_tracker.update_vision(an_c, {"shoulder_pan.pos": 35.0, "shoulder_lift.pos": -10.0}, 12.0))
        self.assertFalse(cont_tracker.update_vision(an_c, {"shoulder_pan.pos": 30.0, "shoulder_lift.pos": -15.0}, 12.3))
        self.assertTrue(cont_tracker.dropped_in_desechos)
        self.assertFalse(cont_tracker.observe_robot_state({"shoulder_pan.pos": 18.0, "shoulder_lift.pos": -40.0}, 12.5))
        # Now dips into Vaso D, then lifts out -> triggers early stop
        self.assertFalse(cont_tracker.observe_robot_state({"shoulder_pan.pos": -1.0, "shoulder_lift.pos": 22.0}, 16.0))
        self.assertTrue(cont_tracker.visited_vaso_d)
        self.assertTrue(cont_tracker.observe_robot_state({"shoulder_pan.pos": -3.0, "shoulder_lift.pos": -35.0}, 18.0))
        self.assertIn("VASO D", cont_tracker.stop_reason or "")

    def test_persistent_session_run_policy_invokes_live_callback_and_stops_early(self) -> None:
        class _DummyPolicy:
            def __init__(self) -> None:
                self.config = type("Cfg", (), {"temporal_ensemble_coeff": None, "n_action_steps": 50})()

            def to(self, _d):
                return self

            def eval(self):
                return self

            def reset(self):
                return None

            def select_action(self, _b):
                import torch

                return torch.zeros((1, 6), dtype=torch.float32)

        class _DummyProc:
            def __call__(self, x):
                return x

            def reset(self):
                return None

        class _DummyRobot:
            def __init__(self, _cfg) -> None:
                self.observation_features = {
                    "shoulder_pan.pos": float,
                    "shoulder_lift.pos": float,
                    "elbow_flex.pos": float,
                    "wrist_flex.pos": float,
                    "wrist_roll.pos": float,
                    "gripper.pos": float,
                    "camera_top": (48, 64, 3),
                    "camera_robot": (48, 64, 3),
                }
                self.action_features = {k: float for k in self.observation_features if k.endswith(".pos")}
                self.sent_actions = 0

            def connect(self, calibrate: bool = False) -> None:
                return None

            def get_observation(self) -> dict:
                return {
                    "shoulder_pan.pos": 0.0,
                    "shoulder_lift.pos": -50.0,
                    "elbow_flex.pos": 20.0,
                    "wrist_flex.pos": 90.0,
                    "wrist_roll.pos": 0.0,
                    "gripper.pos": 2.0,
                    "camera_top": np.zeros((48, 64, 3), dtype=np.uint8),
                    "camera_robot": np.zeros((48, 64, 3), dtype=np.uint8),
                }

            def send_action(self, _action: dict) -> dict:
                self.sent_actions += 1
                return _action

            def disconnect(self) -> None:
                return None

        policies = [_DummyPolicy(), _DummyPolicy()]
        p_it = iter(policies)
        session = supervisor.PersistentTriageSession(
            policy_sano_path=Path("dummy_sano"),
            policy_contaminado_path=Path("dummy_cont"),
            robot_port="/dev/tty.dummy",
            robot_id="bilbao_follower",
            robot_type="so101_follower",
            cam_top_idx=1,
            cam_wrist_idx=0,
            width=64,
            height=48,
            device="cpu",
        )
        with (
            mock.patch("lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained", side_effect=lambda _p: next(p_it)),
            mock.patch("lerobot.policies.factory.make_pre_post_processors", side_effect=lambda *_a, **_kw: (_DummyProc(), _DummyProc())),
            mock.patch("lerobot.robots.so_follower.SOFollower", side_effect=_DummyRobot),
        ):
            session.start()
            live_calls = 0

            def _live_cb(top_bgr: np.ndarray, obs_state: dict, _el: float) -> bool:
                nonlocal live_calls
                live_calls += 1
                self.assertEqual(top_bgr.shape, (48, 64, 3))
                self.assertIn("shoulder_pan.pos", obs_state)
                return live_calls >= 2

            rc = session.run_policy("sano", duration_s=4.0, fps=30, live_frame_cb=_live_cb, live_every_n_ticks=2)
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(live_calls, 2)
            self.assertLess(session.robot.sent_actions, 80)
            session.close()

    def test_move_connected_robot_to_pose_lifts_vertically_before_sweeping_pan(self) -> None:
        class _FakeRobot:
            def __init__(self) -> None:
                self.actions: list[dict[str, float]] = []

            def get_observation(self) -> dict[str, float]:
                return {
                    "shoulder_pan.pos": 42.0,
                    "shoulder_lift.pos": 15.0,
                    "elbow_flex.pos": -20.0,
                    "wrist_flex.pos": 50.0,
                    "wrist_roll.pos": -25.0,
                    "gripper.pos": 2.0,
                }

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.actions.append(dict(action))
                return action

        robot = _FakeRobot()
        supervisor._move_connected_robot_to_pose(robot, duration_s=1.0, fps=20, label="HOME_TEST", lift_first=True)
        # First phase lifts shoulder_lift cleanly below -40 while keeping shoulder_pan near 42.0
        self.assertGreater(len(robot.actions), 10)
        lift_idx = next(i for i, a in enumerate(robot.actions) if a["shoulder_lift.pos"] <= -40.0)
        self.assertAlmostEqual(robot.actions[lift_idx]["shoulder_pan.pos"], 42.0, places=3)
        # Final step arrives at DEFAULT_HOME_POSE
        final = robot.actions[-1]
        for k, v in supervisor.DEFAULT_HOME_POSE.items():
            self.assertAlmostEqual(final[k], v, places=3)

    def test_archive_fail_snapshot_copies_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            hud_path = tmp_dir / "triage_hud_latest.jpg"
            fail_path = tmp_dir / "debug_last_fail.jpg"

            self.assertFalse(supervisor._archive_fail_snapshot(hud_path, fail_path))
            self.assertFalse(fail_path.exists())

            hud_path.write_bytes(b"fake-jpeg-bytes")
            self.assertTrue(supervisor._archive_fail_snapshot(hud_path, fail_path))
            self.assertEqual(fail_path.read_bytes(), b"fake-jpeg-bytes")

    def test_persistent_session_refreshes_post_home_chunks_more_frequently(self) -> None:
        from collections import deque
        import torch

        class _QueuePolicy:
            def __init__(self, chunk_steps: int = 6) -> None:
                self.chunk_steps = chunk_steps
                self.config = type("Cfg", (), {"temporal_ensemble_coeff": None, "n_action_steps": chunk_steps})()
                self._action_queue: deque = deque()
                self.predict_calls = 0

            def reset(self) -> None:
                self._action_queue.clear()

            def to(self, _dev):
                return self

            def eval(self):
                return self

            def select_action(self, _batch: dict) -> torch.Tensor:
                if len(self._action_queue) == 0:
                    self.predict_calls += 1
                    self._action_queue.extend([torch.zeros((1, 6)) for _ in range(self.chunk_steps)])
                return self._action_queue.popleft()

        class _DummyProc:
            def reset(self) -> None:
                pass

            def __call__(self, x):
                return x

        class _DummyRobot:
            def __init__(self, *_a, **_kw) -> None:
                self.observation_features = {k: float for k in supervisor.DEFAULT_HOME_POSE}
                self.action_features = dict(self.observation_features)

            def connect(self, calibrate: bool = False) -> None:
                pass

            def get_observation(self) -> dict:
                return dict(supervisor.DEFAULT_HOME_POSE)

            def send_action(self, action: dict) -> dict:
                return action

            def disconnect(self) -> None:
                pass

        policies = [_QueuePolicy(6), _QueuePolicy(6)]
        p_it = iter(policies)
        session = supervisor.PersistentTriageSession(
            policy_sano_path=Path("s"),
            policy_contaminado_path=Path("c"),
            robot_port="/dev/null",
            robot_id="test",
            robot_type="so101_follower",
            cam_top_idx=1,
            cam_wrist_idx=0,
            device="cpu",
            n_action_steps=6,
            post_home_action_steps=2,
        )
        with (
            mock.patch("lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained", side_effect=lambda _p: next(p_it)),
            mock.patch("lerobot.policies.factory.make_pre_post_processors", side_effect=lambda *_a, **_kw: (_DummyProc(), _DummyProc())),
            mock.patch("lerobot.robots.so_follower.SOFollower", side_effect=_DummyRobot),
        ):
            session.start()
            sano_pol = policies[0]
            sano_pol.predict_calls = 0
            # Run 10 ticks at 100 Hz: Chunk 0 takes 6 ticks (tick 0..5), then Chunk 1 + Chunk 2 take 2 ticks each -> 3 calls total
            tick_target = 10
            seen = 0

            def _stop_after_10(_obs, _el) -> bool:
                nonlocal seen
                seen += 1
                return seen > tick_target

            rc = session.run_policy("sano", duration_s=2.0, fps=200, state_cb=_stop_after_10)
            self.assertEqual(rc, 0)
            self.assertEqual(sano_pol.predict_calls, 3)
            session.close()

    def test_persistent_session_sets_per_policy_action_steps_and_retries_camera_timeout(self) -> None:
        from collections import deque
        from types import SimpleNamespace
        import torch

        class _CfgPolicy:
            def __init__(self) -> None:
                self.config = SimpleNamespace(temporal_ensemble_coeff=None, n_action_steps=100)
                self._action_queue = deque(maxlen=100)

            def to(self, _dev):
                return self

            def eval(self):
                return self

            def reset(self) -> None:
                self._action_queue.clear()

            def select_action(self, _batch: dict) -> torch.Tensor:
                return torch.zeros((1, 6))

        class _DummyProc:
            def reset(self) -> None:
                pass

            def __call__(self, x):
                return x

        class _FlakyCam:
            def __init__(self) -> None:
                self.calls = 0

            def async_read(self, timeout_ms: int = 200):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("transient timeout")
                return np.zeros((480, 640, 3), dtype=np.uint8)

        class _DummyRobot:
            def __init__(self, *_a, **_kw) -> None:
                self.observation_features = {k: float for k in supervisor.DEFAULT_HOME_POSE}
                self.action_features = dict(self.observation_features)
                self.cameras = {"camera_top": _FlakyCam()}

            def connect(self, calibrate: bool = False) -> None:
                pass

            def get_observation(self) -> dict:
                return dict(supervisor.DEFAULT_HOME_POSE)

            def disconnect(self) -> None:
                pass

        policies = [_CfgPolicy(), _CfgPolicy()]
        p_it = iter(policies)
        session = supervisor.PersistentTriageSession(
            policy_sano_path=Path("s"),
            policy_contaminado_path=Path("c"),
            robot_port="/dev/null",
            robot_id="test",
            robot_type="so101_follower",
            cam_top_idx=1,
            cam_wrist_idx=0,
            device="cpu",
            n_action_steps=50,
            n_action_steps_contaminado=35,
        )
        with (
            mock.patch("lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained", side_effect=lambda _p: next(p_it)),
            mock.patch("lerobot.policies.factory.make_pre_post_processors", side_effect=lambda *_a, **_kw: (_DummyProc(), _DummyProc())),
            mock.patch("lerobot.robots.so_follower.SOFollower", side_effect=_DummyRobot),
        ):
            session.start()
            self.assertEqual(policies[0].config.n_action_steps, 50)
            self.assertEqual(policies[1].config.n_action_steps, 35)
            frame = session.capture_top_bgr(max_attempts=3)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertGreaterEqual(session.robot.cameras["camera_top"].calls, 3)
            session.close()

    def test_early_stop_contaminado_handles_merged_cubes_in_desechos(self) -> None:
        baseline = {
            "verdes_en_A": 0,
            "rojos_en_A": 1,
            "verdes_en_B_SUP": 2,
            "rojos_en_B_INF": 1,
            "rojos_area_en_B_INF": 520.0,
        }
        tracker = supervisor.EarlyStopTracker(target_key="contaminado", baseline_counts=baseline)
        # Second red cube drops touching the first in ZONA_B_INF -> contour count stays 1, but area doubles
        merged_an = {
            "verdes_en_A": 0,
            "rojos_en_A": 0,
            "verdes_en_B_SUP": 2,
            "rojos_en_B_INF": 1,
            "rojos_area_en_B_INF": 1040.0,
        }
        self.assertFalse(tracker.update_vision(merged_an, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": 10.0}, 11.0))
        self.assertFalse(tracker.dropped_in_desechos)
        self.assertFalse(tracker.update_vision(merged_an, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": 10.0}, 12.0))
        self.assertTrue(tracker.dropped_in_desechos)
        self.assertFalse(tracker.observe_robot_state({"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 15.0}, 14.0))
        self.assertTrue(tracker.observe_robot_state({"shoulder_pan.pos": -4.0, "shoulder_lift.pos": -35.0}, 16.0))

    def test_parse_args_defaults_use_robust_contaminado_config(self) -> None:
        args = supervisor.parse_args(["--mode", "run"])
        self.assertEqual(args.duration_contaminado, 27.0)
        self.assertEqual(args.n_action_steps, 50)
        self.assertEqual(args.n_action_steps_contaminado, 60)
        self.assertEqual(args.post_home_steps, 0)

    def test_polygon_zona_a_treats_cubes_outside_curved_tape_as_lost(self) -> None:
        cfg = supervisor.load_vision_config(Path("outputs/triage_vision_config.json"))
        # Frame 640x480 with:
        # 1) red cube inside curved ZONA_A cross (440, 125)
        # 2) red cube pushed outside curved tape bottom-right (505, 198) -> inside bounding box [328,42,551,247], OUTSIDE tape arc
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        cv2.rectangle(frame, (425, 110), (455, 140), (0, 0, 220), -1)
        cv2.rectangle(frame, (490, 183), (520, 213), (0, 0, 220), -1)

        an = supervisor.analyze_frame(frame, cfg)
        self.assertEqual(an["rojos_en_A"], 1)
        rois_by_x = {c.centroid[0]: c.roi for c in an["cubes"]}
        self.assertEqual(rois_by_x[440], "ZONA_A")
        self.assertIsNone(rois_by_x[505])

    def test_supervisor_marks_lost_cube_outside_tape_and_continues_to_next_cube(self) -> None:
        # Cycle 1 starts with 2 red cubes in A -> cube 1 knocked outside tape (rojos_en_A: 2->1, rojos_en_B_INF unchanged 0->0)
        # -> Cycle 2 picks remaining red cube (1->0, rojos_en_B_INF: 0->1) -> Cycle 3 completed!
        scenes = [
            {"verdes_en_A": 0, "rojos_en_A": 2, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 0, "target": "contaminado", "next_order": "R", "cubes": []},
            {"verdes_en_A": 0, "rojos_en_A": 1, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 0, "target": "contaminado", "next_order": "R", "cubes": []},
            {"verdes_en_A": 0, "rojos_en_A": 1, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 0, "target": "contaminado", "next_order": "R", "cubes": []},
            {"verdes_en_A": 0, "rojos_en_A": 0, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 1, "target": "completed", "next_order": "DONE", "cubes": []},
            {"verdes_en_A": 0, "rojos_en_A": 0, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 1, "target": "completed", "next_order": "DONE", "cubes": []},
        ]
        it = iter(scenes)
        cfg = supervisor.default_vision_config()
        rc = supervisor.run_autonomous_supervisor(
            config=cfg,
            policy_sano=Path(__file__),
            policy_contaminado=Path(__file__),
            robot_port="/dev/null",
            robot_id="test",
            robot_type="so101_follower",
            cam_wrist_idx=0,
            max_retries=0,  # Proves lost cube does NOT burn retries or abort!
            scanner_fn=lambda: next(it),
            runner_fn=lambda _cmd: 0,
        )
        self.assertEqual(rc, 0)

    def test_early_stop_tracks_lost_red_cube_outside_tape_and_awaits_vaso_d(self) -> None:
        baseline = {
            "verdes_en_A": 0,
            "rojos_en_A": 1,
            "verdes_en_B_SUP": 2,
            "rojos_en_B_INF": 1,
            "rojos_perdidos": 0,
        }
        tracker = supervisor.EarlyStopTracker(target_key="contaminado", baseline_counts=baseline)
        lost_an = {
            "verdes_en_A": 0,
            "rojos_en_A": 0,
            "verdes_en_B_SUP": 2,
            "rojos_en_B_INF": 1,
            "rojos_perdidos": 1,
        }
        # Stable 2 frames after 10s marks dropped/lost so observe_robot_state completes cleanly right after VASO D dip
        self.assertFalse(tracker.update_vision(lost_an, {"shoulder_pan.pos": -30.0, "shoulder_lift.pos": 10.0}, 11.0))
        self.assertFalse(tracker.update_vision(lost_an, {"shoulder_pan.pos": -20.0, "shoulder_lift.pos": 0.0}, 11.5))
        self.assertTrue(tracker.dropped_in_desechos)
        self.assertTrue(tracker.lost_outside_tape)
        self.assertFalse(tracker.observe_robot_state({"shoulder_pan.pos": -1.5, "shoulder_lift.pos": 48.0}, 13.0))
        self.assertTrue(tracker.visited_vaso_d)
        self.assertTrue(tracker.observe_robot_state({"shoulder_pan.pos": -4.0, "shoulder_lift.pos": -35.0}, 15.0))

    def test_early_stop_contaminado_cuts_empty_miss_fast_and_parses_post_home_contam(self) -> None:
        args = supervisor.parse_args(["--mode", "run"])
        self.assertEqual(args.post_home_steps_contaminado, 25)
        tracker = supervisor.EarlyStopTracker(
            target_key="contaminado",
            baseline_counts={"verdes_en_A": 0, "rojos_en_A": 1, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 1},
        )
        miss_an = {"verdes_en_A": 0, "rojos_en_A": 1, "verdes_en_B_SUP": 2, "rojos_en_B_INF": 1}
        self.assertFalse(tracker.update_vision(miss_an, {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -20.0}, 11.0))
        self.assertFalse(tracker.update_vision(miss_an, {"shoulder_pan.pos": -4.0, "shoulder_lift.pos": -10.0}, 11.4))
        self.assertTrue(tracker.update_vision(miss_an, {"shoulder_pan.pos": -3.0, "shoulder_lift.pos": -15.0}, 11.8))
        self.assertIn("Agarre vacío", tracker.stop_reason or "")
