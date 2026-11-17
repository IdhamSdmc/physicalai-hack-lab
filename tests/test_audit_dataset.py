"""Validation tests for the LeRobot Parquet dataset auditor (`scripts/02_audit_dataset.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests._loader import REPO_ROOT, load_script_module

audit_mod = load_script_module("02_audit_dataset.py", "audit_dataset")


def _make_smooth_episode(
    ep_idx: int,
    n_frames: int = 120,
    fps: float = 30.0,
    init_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n_frames, dtype=np.float32) / float(fps)
    phase = np.linspace(0.0, 2.0 * np.pi, n_frames, dtype=np.float32)
    states = np.stack(
        [
            init_offset + 15.0 * np.sin(phase),
            -40.0 + 10.0 * np.sin(phase),
            60.0 + 12.0 * np.cos(phase),
            30.0 + 5.0 * np.sin(phase),
            -10.0 + 8.0 * np.cos(phase),
            1.4 + 12.0 * np.sin(phase * 0.5) ** 2,
        ],
        axis=1,
    ).astype(np.float32)
    actions = states.copy()
    episodes = np.full(n_frames, ep_idx, dtype=np.int64)
    frame_indices = np.arange(n_frames, dtype=np.int64)
    return episodes, frame_indices, t, states, actions


def _write_synthetic_dataset(
    root: Path,
    episode_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    fps: int = 30,
    shuffle_rows: bool = False,
) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "robot_type": "so101_follower",
        "codebase_version": "v3.0",
        "fps": fps,
        "features": {
            "observation.state": {"shape": [6]},
            "action": {"shape": [6]},
            "observation.images.top": {"shape": [480, 640, 3]},
            "observation.images.wrist": {"shape": [480, 640, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick red cube"}) + "\n")

    ep_col = np.concatenate([x[0] for x in episode_arrays])
    fr_col = np.concatenate([x[1] for x in episode_arrays])
    ts_col = np.concatenate([x[2] for x in episode_arrays])
    st_col = np.concatenate([x[3] for x in episode_arrays], axis=0)
    ac_col = np.concatenate([x[4] for x in episode_arrays], axis=0)

    if shuffle_rows:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(ep_col))
        ep_col = ep_col[perm]
        fr_col = fr_col[perm]
        ts_col = ts_col[perm]
        st_col = st_col[perm]
        ac_col = ac_col[perm]

    table = pa.table(
        {
            "episode_index": pa.array(ep_col, type=pa.int64()),
            "frame_index": pa.array(fr_col, type=pa.int64()),
            "timestamp": pa.array(ts_col, type=pa.float32()),
            "observation.state": pa.array([row.tolist() for row in st_col], type=pa.list_(pa.float32())),
            "action": pa.array([row.tolist() for row in ac_col], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


class AuditDatasetTests(unittest.TestCase):
    def test_clean_dataset_passes_with_zero_flagged_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eps = [_make_smooth_episode(i, init_offset=float(i) * 0.4) for i in range(3)]
            _write_synthetic_dataset(root, eps, shuffle_rows=True)

            report = audit_mod.audit_dataset_root(root, repo_id="team/clean_so101")
            self.assertEqual(report["total_episodes"], 3)
            self.assertEqual(report["clean_episodes"], 3)
            self.assertEqual(report["flagged_count"], 0)
            self.assertEqual(report["discard_rate_pct"], 0.0)
            self.assertFalse(report["initial_sensor_freeze"])
            self.assertEqual(report["tasks"], ["pick red cube"])
            self.assertEqual(report["schema_warnings"], [])

    def test_detects_fps_drop_hesitation_leading_idle_and_jerk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep0 = _make_smooth_episode(0, init_offset=0.1)

            # Ep 1: caida de FPS (salto de 120 ms)
            e1, f1, t1, s1, a1 = _make_smooth_episode(1, init_offset=0.2)
            t1[50:] += 0.12
            ep1 = (e1, f1, t1, s1, a1)

            # Ep 2: hesitation > 45% (65% quieto al final)
            e2, f2, t2, s2, a2 = _make_smooth_episode(2, init_offset=0.3)
            s2[40:] = s2[40]
            ep2 = (e2, f2, t2, s2, a2)

            # Ep 3: tiempo muerto inicial > 1.5s (75 frames = 2.5s quieto al inicio)
            e3, f3, t3, s3, a3 = _make_smooth_episode(3, n_frames=180, init_offset=0.4)
            s3[:75] = s3[75]
            ep3 = (e3, f3, t3, s3, a3)

            # Ep 4: tiron brusco (jerk alto en acciones)
            e4, f4, t4, s4, a4 = _make_smooth_episode(4, init_offset=0.5)
            a4[60] += 85.0
            ep4 = (e4, f4, t4, s4, a4)

            _write_synthetic_dataset(root, [ep0, ep1, ep2, ep3, ep4])
            report = audit_mod.audit_dataset_root(root, repo_id="team/mixed_so101")

            self.assertEqual(report["total_episodes"], 5)
            self.assertEqual(report["clean_episodes"], 1)
            self.assertEqual(report["flagged_ids"], [1, 2, 3, 4])

            reasons_by_ep = {item["episode"]: " | ".join(item["reasons"]) for item in report["flagged_episodes"]}
            self.assertIn("caídas de FPS", reasons_by_ep[1])
            self.assertIn("Hesitation alta", reasons_by_ep[2])
            self.assertIn("Tiempo muerto inicial", reasons_by_ep[3])
            self.assertIn("Tirones bruscos", reasons_by_ep[4])

    def test_detects_bit_identical_sensor_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eps = [_make_smooth_episode(i, init_offset=0.0) for i in range(3)]
            _write_synthetic_dataset(root, eps)

            report = audit_mod.audit_dataset_root(root, repo_id="team/frozen_init")
            self.assertTrue(report["initial_sensor_freeze"])

    def test_cli_json_report_and_full_delete_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_root = tmp_path / "dataset"
            json_out = tmp_path / "audit_report.json"

            bad_eps = []
            for i in range(12):
                e, f, t, s, a = _make_smooth_episode(i, init_offset=float(i) * 0.2)
                s[:] = s[0]  # 100% hesitation -> todos marcados
                bad_eps.append((e, f, t, s, a))
            _write_synthetic_dataset(ds_root, bad_eps)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "02_audit_dataset.py"),
                    "--repo-id",
                    "team/bad12",
                    "--local-dir",
                    str(ds_root),
                    "--json-out",
                    str(json_out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(json_out.exists())
            parsed = json.loads(json_out.read_text())
            self.assertEqual(parsed["flagged_ids"], list(range(12)))
            self.assertEqual(parsed["discard_rate_pct"], 100.0)
            self.assertIn("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]", proc.stdout)
            self.assertIn(f"--root {ds_root}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
