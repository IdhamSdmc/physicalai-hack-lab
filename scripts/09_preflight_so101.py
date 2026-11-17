#!/usr/bin/env python3
"""
================================================================================
PASO 0.5 / PREFLIGHT — VALIDADOR INTEGRAL SO-101 (GUÍA HACKATHON BILBAO)
================================================================================
Verifica antes de tocar el robot (y en la mesa el viernes en La Perrera) todas
las condiciones críticas que documenta https://hackathonbilbao.eus/guia-so-101:
  1. Variables en `scripts/env_hackathon.sh` (puertos distintos, IDs distintos,
     cámaras sin índices duplicados, nombres `rollout_*` para lerobot-rollout).
  2. Calibraciones SO-101 en `~/.cache/huggingface/lerobot/calibration/`
     (6 motores Feetech STS3215, rangos válidos y sin duplicado Leader=Follower).
  3. Contrato real de CLI (`RecordConfig` y `RolloutConfig` base/episodic/dagger)
     directamente contra el parser de LeRobot sin requerir hardware conectado.
  4. Opcional (`--check-hardware`): existencia física de puertos USB y cámaras.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import yaml

SO101_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CALIBRATION_FIELDS: tuple[str, ...] = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
VALID_FOLLOWER_TYPES = ("so101_follower", "so100_follower")
VALID_TELEOP_TYPES = ("so101_leader", "so100_leader")


def load_env_from_script(env_script: Path) -> dict[str, str]:
    """Source a bash env file in a subshell and return resulting environment variables."""
    if not env_script.exists():
        raise FileNotFoundError(f"Archivo de entorno no encontrado: {env_script}")
    cmd = [
        "bash",
        "-c",
        f'HACK_ENV_QUIET=1 source "{env_script.resolve()}" && "{sys.executable}" -c "import os, json; print(json.dumps(dict(os.environ)))"',
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def validate_camera_spec(cameras_raw: str) -> tuple[list[str], list[str], dict[str, Any]]:
    """Parse and validate `ROBOT_CAMERAS` YAML/JSON dictionary syntax and USB constraints."""
    errors: list[str] = []
    warnings: list[str] = []
    if not cameras_raw.strip():
        return ["ROBOT_CAMERAS está vacío."], warnings, {}

    try:
        parsed = yaml.safe_load(cameras_raw)
    except yaml.YAMLError as exc:
        return [f"ROBOT_CAMERAS no es YAML/JSON válido: {exc}"], warnings, {}

    if not isinstance(parsed, dict) or not parsed:
        return ["ROBOT_CAMERAS debe ser un diccionario no vacío de cámaras."], warnings, {}

    seen_indices: dict[str, str] = {}
    for cam_name, cfg in parsed.items():
        if not isinstance(cfg, dict):
            errors.append(f"Cámara '{cam_name}' debe ser un objeto de configuración.")
            continue
        if "index_or_path" not in cfg:
            errors.append(f"Cámara '{cam_name}' no define 'index_or_path'.")
            continue
        idx_key = str(cfg["index_or_path"])
        if idx_key in seen_indices:
            errors.append(
                f"Índice/ruta de cámara duplicado ({idx_key}) entre '{seen_indices[idx_key]}' y '{cam_name}'."
            )
        else:
            seen_indices[idx_key] = cam_name

        w = int(cfg.get("width", 0))
        h = int(cfg.get("height", 0))
        fps = float(cfg.get("fps", 0))
        if w > 640 or h > 480:
            warnings.append(
                f"Cámara '{cam_name}' a {w}x{h}: resoluciones >640x480 pueden saturar el bus USB 2.0."
            )
        if fps and abs(fps - 30.0) > 1.0:
            warnings.append(f"Cámara '{cam_name}' con fps={fps} (guía SO-101 recomienda 30 FPS).")

    return errors, warnings, parsed


def _repo_leaf(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].strip()


def validate_hackathon_env(env: dict[str, str]) -> dict[str, Any]:
    """Validate all SO-101 environment invariants against the Bilbao guide & LeRobot v0.7.0."""
    errors: list[str] = []
    warnings: list[str] = []

    robot_type = env.get("ROBOT_TYPE", "").strip()
    teleop_type = env.get("TELEOP_TYPE", "").strip()
    robot_id = env.get("ROBOT_ID", "").strip()
    teleop_id = env.get("TELEOP_ID", "").strip()
    follower_port = env.get("FOLLOWER_PORT", "").strip()
    leader_port = env.get("LEADER_PORT", "").strip()
    dataset_repo = env.get("DATASET_REPO", "").strip()
    rollout_repo = env.get("ROLLOUT_DATASET_REPO", "").strip()
    dagger_repo = env.get("DAGGER_DATASET_REPO", "").strip()

    if robot_type not in VALID_FOLLOWER_TYPES:
        errors.append(f"ROBOT_TYPE inválido: '{robot_type}' (esperado: {VALID_FOLLOWER_TYPES}).")
    if teleop_type not in VALID_TELEOP_TYPES:
        errors.append(f"TELEOP_TYPE inválido: '{teleop_type}' (esperado: {VALID_TELEOP_TYPES}).")

    if not robot_id or not teleop_id:
        errors.append("ROBOT_ID y TELEOP_ID son obligatorios para guardar calibraciones separadas.")
    elif robot_id == teleop_id:
        errors.append(
            f"ROBOT_ID == TELEOP_ID ('{robot_id}'): en LeRobot esto mezcla calibraciones y el brazo se mueve mal sin dar error."
        )

    if not follower_port or not leader_port:
        errors.append("FOLLOWER_PORT y LEADER_PORT son obligatorios.")
    elif follower_port == leader_port:
        errors.append(f"FOLLOWER_PORT y LEADER_PORT apuntan al mismo puerto ('{follower_port}').")

    if not dataset_repo:
        errors.append("DATASET_REPO está vacío.")
    elif _repo_leaf(dataset_repo).startswith("eval_"):
        errors.append(
            f"DATASET_REPO ('{dataset_repo}') empieza por 'eval_', lo cual rechaza lerobot-record."
        )

    for var_name, repo_val in (("ROLLOUT_DATASET_REPO", rollout_repo), ("DAGGER_DATASET_REPO", dagger_repo)):
        if not repo_val:
            errors.append(f"{var_name} está vacío (requerido para lerobot-rollout con grabación).")
        elif not _repo_leaf(repo_val).startswith("rollout_"):
            errors.append(
                f"{var_name} ('{repo_val}') debe empezar por 'rollout_' tras '<user>/' en LeRobot v0.7.0."
            )

    cam_errors, cam_warnings, parsed_cams = validate_camera_spec(env.get("ROBOT_CAMERAS", ""))
    errors.extend(cam_errors)
    warnings.extend(cam_warnings)

    return {
        "errors": errors,
        "warnings": warnings,
        "parsed_cameras": parsed_cams,
    }


def validate_calibration_file(fpath: Path) -> tuple[list[str], dict[str, Any] | None]:
    """Validate one LeRobot SO-101 calibration JSON file structure and joint ranges."""
    if not fpath.is_file():
        return [f"No existe archivo de calibración: {fpath}"], None

    try:
        data = json.loads(fpath.read_text())
    except json.JSONDecodeError as exc:
        return [f"JSON inválido en {fpath}: {exc}"], None

    if not isinstance(data, dict):
        return [f"Formato inválido en {fpath}: se esperaba objeto JSON por articulación."], None

    errors: list[str] = []
    for expected_idx, motor_name in enumerate(SO101_MOTORS, start=1):
        entry = data.get(motor_name)
        if not isinstance(entry, dict):
            errors.append(f"{fpath.name}: falta articulación '{motor_name}'.")
            continue
        for field_name in CALIBRATION_FIELDS:
            if field_name not in entry:
                errors.append(f"{fpath.name} [{motor_name}]: falta campo '{field_name}'.")
        if "id" in entry and int(entry["id"]) != expected_idx:
            errors.append(
                f"{fpath.name} [{motor_name}]: motor id={entry['id']} (esperado {expected_idx}; revisa IDs de servos)."
            )
        if "range_min" in entry and "range_max" in entry:
            if float(entry["range_min"]) >= float(entry["range_max"]):
                errors.append(
                    f"{fpath.name} [{motor_name}]: range_min ({entry['range_min']}) >= range_max ({entry['range_max']})."
                )
    return errors, data


def validate_calibration_pair(
    calibration_dir: Path,
    robot_id: str,
    teleop_id: str,
    require_files: bool = False,
) -> dict[str, Any]:
    """Validate SO-101 follower and leader calibration files and check against accidental swaps."""
    follower_path = calibration_dir / "robots" / "so_follower" / f"{robot_id}.json"
    leader_path = calibration_dir / "teleoperators" / "so_leader" / f"{teleop_id}.json"

    errors: list[str] = []
    warnings: list[str] = []

    if not follower_path.exists() and not require_files:
        warnings.append(f"Follower sin calibrar todavía ({follower_path}); ejecuta `lerobot-calibrate` en la mesa.")
        follower_data = None
    else:
        f_errs, follower_data = validate_calibration_file(follower_path)
        errors.extend(f_errs)

    if not leader_path.exists() and not require_files:
        warnings.append(f"Leader sin calibrar todavía ({leader_path}); ejecuta `lerobot-calibrate` en la mesa.")
        leader_data = None
    else:
        l_errs, leader_data = validate_calibration_file(leader_path)
        errors.extend(l_errs)

    if follower_data is not None and leader_data is not None and follower_data == leader_data:
        errors.append(
            f"Calibración idéntica entre Follower ({follower_path.name}) y Leader ({leader_path.name}). "
            "Recalibra cada brazo por separado con su propio --robot.id / --teleop.id."
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "follower_path": str(follower_path),
        "leader_path": str(leader_path),
    }


def parse_lerobot_config(config_cls: type[Any], cli_args: list[str]) -> Any:
    """Parse a LeRobot dataclass config mirroring `@lerobot.configs.parser.wrap()`."""
    import draccus

    from lerobot.configs import parser as lerobot_parser
    from lerobot.utils.utils import has_method

    prev_argv = sys.argv
    sys.argv = ["lerobot-preflight", *cli_args]
    try:
        filtered = list(cli_args)
        if has_method(config_cls, "__get_path_fields__"):
            filtered = lerobot_parser.filter_path_args(config_cls.__get_path_fields__(), filtered)
        return draccus.parse(config_cls, args=filtered)
    finally:
        sys.argv = prev_argv


def validate_lerobot_cli_contracts(env: dict[str, str]) -> list[str]:
    """Dry-run LeRobot parser contracts for RecordConfig and RolloutConfig without hardware."""
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.scripts.lerobot_record import RecordConfig

    errors: list[str] = []
    robot_type = env["ROBOT_TYPE"]
    teleop_type = env["TELEOP_TYPE"]
    follower_port = env["FOLLOWER_PORT"]
    leader_port = env["LEADER_PORT"]
    robot_id = env["ROBOT_ID"]
    teleop_id = env["TELEOP_ID"]
    cameras = env["ROBOT_CAMERAS"]
    dataset_repo = env["DATASET_REPO"]
    rollout_repo = env["ROLLOUT_DATASET_REPO"]
    dagger_repo = env["DAGGER_DATASET_REPO"]
    task_desc = env.get("TASK_DESC", "pick and place")
    device_name = env.get("HACK_DEVICE", "cpu") or "cpu"

    try:
        parse_lerobot_config(
            RecordConfig,
            [
                f"--robot.type={robot_type}",
                f"--robot.port={follower_port}",
                f"--robot.id={robot_id}",
                f"--robot.cameras={cameras}",
                f"--teleop.type={teleop_type}",
                f"--teleop.port={leader_port}",
                f"--teleop.id={teleop_id}",
                f"--dataset.repo_id={dataset_repo}",
                "--dataset.num_episodes=5",
                f"--dataset.single_task={task_desc}",
                "--dataset.push_to_hub=false",
                "--dataset.no_stamp=true",
            ],
        )
    except Exception as exc:
        errors.append(f"RecordConfig rechazó los parámetros actuales: {exc}")

    # Creamos un config.json mínimo de ACT temporal para validar RolloutConfig sin checkpoint real
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_dir = Path(tmp_dir)
        (policy_dir / "config.json").write_text(
            json.dumps(
                {
                    "type": "act",
                    "chunk_size": 50,
                    "n_action_steps": 50,
                    "device": "cpu",
                }
            )
        )

        rollout_cases = [
            (
                "base",
                [
                    "--strategy.type=base",
                    f"--policy.path={policy_dir}",
                    f"--device={device_name}",
                    f"--robot.type={robot_type}",
                    f"--robot.port={follower_port}",
                    f"--robot.id={robot_id}",
                    f"--robot.cameras={cameras}",
                    f"--task={task_desc}",
                    "--duration=60",
                ],
            ),
            (
                "episodic",
                [
                    "--strategy.type=episodic",
                    f"--policy.path={policy_dir}",
                    f"--device={device_name}",
                    f"--robot.type={robot_type}",
                    f"--robot.port={follower_port}",
                    f"--robot.id={robot_id}",
                    f"--robot.cameras={cameras}",
                    f"--teleop.type={teleop_type}",
                    f"--teleop.port={leader_port}",
                    f"--teleop.id={teleop_id}",
                    f"--dataset.repo_id={rollout_repo}",
                    "--dataset.num_episodes=5",
                    f"--dataset.single_task={task_desc}",
                    "--dataset.push_to_hub=false",
                ],
            ),
            (
                "dagger",
                [
                    "--strategy.type=dagger",
                    f"--policy.path={policy_dir}",
                    f"--device={device_name}",
                    f"--robot.type={robot_type}",
                    f"--robot.port={follower_port}",
                    f"--robot.id={robot_id}",
                    f"--robot.cameras={cameras}",
                    f"--teleop.type={teleop_type}",
                    f"--teleop.port={leader_port}",
                    f"--teleop.id={teleop_id}",
                    f"--dataset.repo_id={dagger_repo}",
                    f"--dataset.single_task={task_desc}",
                    "--dataset.push_to_hub=false",
                    "--duration=60",
                ],
            ),
        ]

        for label, cli_args in rollout_cases:
            try:
                cfg = parse_lerobot_config(RolloutConfig, cli_args)
                if cfg.dataset is not None:
                    leaf = _repo_leaf(cfg.dataset.repo_id)
                    if not leaf.startswith("rollout_"):
                        errors.append(
                            f"RolloutConfig ({label}): dataset '{cfg.dataset.repo_id}' requiere prefijo 'rollout_'."
                        )
            except Exception as exc:
                errors.append(f"RolloutConfig ({label}) rechazó los parámetros actuales: {exc}")

    return errors


def check_hardware_devices(env: dict[str, str], parsed_cams: dict[str, Any]) -> list[str]:
    """Check physical serial port presence and OpenCV camera openability."""
    errors: list[str] = []
    for key in ("FOLLOWER_PORT", "LEADER_PORT"):
        port_path = Path(env[key])
        if not port_path.exists():
            errors.append(f"Puerto físico no encontrado ({key}={port_path}). Ejecuta `lerobot-find-port`.")

    try:
        import cv2
    except ImportError:
        errors.append("OpenCV (cv2) no está instalado para comprobar cámaras.")
        return errors

    for cam_name, cfg in parsed_cams.items():
        idx = cfg["index_or_path"]
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            errors.append(f"No se pudo abrir la cámara '{cam_name}' (index_or_path={idx}).")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight completo SO-101 para Hackathon LeRobot Bilbao.")
    parser.add_argument(
        "--env-file",
        type=str,
        default="scripts/env_hackathon.sh",
        help="Ruta al script de variables de entorno (por defecto scripts/env_hackathon.sh).",
    )
    parser.add_argument(
        "--use-current-env",
        action="store_true",
        help="Leer variables de os.environ actual en lugar de cargar --env-file.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=str(Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"),
        help="Directorio raíz de calibración LeRobot.",
    )
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help="Fallar si aún no existen archivos JSON de calibración.",
    )
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="Comprobar puertos USB físicos y apertura de cámaras OpenCV.",
    )
    parser.add_argument(
        "--skip-cli-parse",
        action="store_true",
        help="Omitir validación draccus contra RecordConfig/RolloutConfig.",
    )
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  PREFLIGHT SO-101 — HACKATHON LEROBOT BILBAO (LA PERRERA)")
    print("=" * 78)

    if args.use_current_env:
        env = dict(os.environ)
    else:
        try:
            env = load_env_from_script(Path(args.env_file))
        except Exception as exc:
            print(f"[ERROR] No se pudo cargar {args.env_file}: {exc}")
            return 1

    env_result = validate_hackathon_env(env)
    cal_result = validate_calibration_pair(
        Path(args.calibration_dir),
        robot_id=env.get("ROBOT_ID", ""),
        teleop_id=env.get("TELEOP_ID", ""),
        require_files=args.require_calibration or args.check_hardware,
    )

    all_errors: list[str] = [*env_result["errors"], *cal_result["errors"]]
    all_warnings: list[str] = [*env_result["warnings"], *cal_result["warnings"]]

    if not args.skip_cli_parse and not env_result["errors"]:
        all_errors.extend(validate_lerobot_cli_contracts(env))

    if args.check_hardware:
        all_errors.extend(check_hardware_devices(env, env_result["parsed_cameras"]))

    print(f"[•] Robot / Teleop     : {env.get('ROBOT_TYPE')} ({env.get('ROBOT_ID')}) / {env.get('TELEOP_TYPE')} ({env.get('TELEOP_ID')})")
    print(f"[•] Puertos            : follower={env.get('FOLLOWER_PORT')} | leader={env.get('LEADER_PORT')}")
    print(f"[•] Cámaras            : {list(env_result['parsed_cameras'].keys())}")
    print(f"[•] Datasets           : record={env.get('DATASET_REPO')} | rollout={env.get('ROLLOUT_DATASET_REPO')} | dagger={env.get('DAGGER_DATASET_REPO')}")
    print(f"[•] Calibración rutas  : {cal_result['follower_path']} | {cal_result['leader_path']}")

    if all_warnings:
        print("\n--- Avisos ---")
        for w in all_warnings:
            print(f"  [!] {w}")

    if all_errors:
        print("\n--- ERRORES CRÍTICOS DETECTADOS ---")
        for err in all_errors:
            print(f"  [X] {err}")
        print("=" * 78)
        return 1

    print("\n✅ [PREFLIGHT OK] Configuración, contratos CLI v0.7.0 y reglas SO-101 validados.")
    print("   Recordatorios físicos de la guía oficial antes de dar corriente:")
    print("   1. Etiqueta las fuentes (5V Leader vs 12V Follower) y quita corriente antes de tocar cables.")
    print("   2. Conecta al menos 1 cámara directa al portátil (no ambas en un hub USB 2.0 sin alimentar).")
    if platform.system() == "Darwin":
        print("   3. macOS: da permiso de 'Monitorización de entrada' a Terminal para teclas de grabar/DAgger.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
