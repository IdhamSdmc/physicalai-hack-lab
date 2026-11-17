#!/usr/bin/env python3
"""
================================================================================
SPRINT 2 (PARTE B) — ANATOMÍA DEL CHECKPOINT E INFERENCIA REAL EN APPLE M5
================================================================================
[¿PARA QUÉ SIRVE ESTE SCRIPT?]
Acabas de entrenar tu primer modelo ACT (`outputs/train/.../pretrained_model`).
Pero en robótica una red neuronal no escupe directamente grados de motor:
  1. Primero normaliza las entradas (`mean=0, std=1`) con `policy_preprocessor`.
  2. Luego la red ACT (`model.safetensors`, 197 MB) predice un bloque de 50 pasos
     futuros en espacio normalizado.
  3. Finalmente `policy_postprocessor` DES-NORMALIZA la predicción y la convierte
     de nuevo en grados físicos reales (`[-100°, +100°]`) para mover los 6 servos
     Feetech del brazo SO-101.

Este script carga tu checkpoint recién entrenado en el M5, le pasa un frame real
del brazo SO-101 y compara la predicción del modelo contra lo que hizo el humano.
================================================================================
"""

import argparse
import glob
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


def extract_action_tensor(policy_output: Any) -> torch.Tensor:
    """Handle both bare Tensor (ACT) and LeRobot v0.7.0 PolicyOutput dict."""
    if isinstance(policy_output, dict):
        if "action" in policy_output:
            return policy_output["action"]
        raise KeyError(f"PolicyOutput dict missing 'action' key: {list(policy_output.keys())}")
    return policy_output


def resolve_checkpoint_dir(ckpt_dir: str | None) -> Path:
    """Resolve an explicit checkpoint path or locate the latest local SO-101 checkpoint."""
    if ckpt_dir:
        p = Path(ckpt_dir)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {p}")
        return p

    ckpts = sorted(glob.glob("outputs/train/act_so101*/checkpoints/*/pretrained_model"))
    if not ckpts:
        raise FileNotFoundError(
            "No se encontró ningún checkpoint en outputs/train/. Ejecuta primero 03_train_local_m5.sh"
        )
    return Path(ckpts[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspecciona un checkpoint ACT y valida inferencia vs ground truth.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Ruta a pretrained_model/")
    parser.add_argument("--repo-id", type=str, default="lerobot/svla_so101_pickplace", help="Dataset de prueba")
    parser.add_argument("--local-dir", type=str, default=None, help="Ruta local opcional al dataset (modo offline)")
    parser.add_argument("--episode", type=int, default=0, help="Índice de episodio")
    parser.add_argument("--frame", type=int, default=10, help="Frame dentro del episodio")
    parser.add_argument("--device", type=str, default=None, help="Forzar dispositivo (mps, cuda, cpu)")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  PASO 3: INSPECCIÓN DE CHECKPOINT E INFERENCIA REAL EN EL BRAZO SO-101")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [MARCA 1: LOCALIZAR EL ÚLTIMO CHECKPOINT ENTRENADO EN DISCO]
    # -------------------------------------------------------------------------
    try:
        ckpt_dir = resolve_checkpoint_dir(args.ckpt_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"\n[MARCA 1] Checkpoint detectado : {ckpt_dir}")
    for f in sorted(ckpt_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{f.stat().st_size / 1024:.1f} KB"
        print(f"          • {f.name:<62} ({size_str})")

    # -------------------------------------------------------------------------
    # [MARCA 2: CARGAR POLÍTICA ACT + PREPROCESSOR + POSTPROCESSOR]
    # -------------------------------------------------------------------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[MARCA 2] Cargando pesos en dispositivo: {device}...")

    policy = ACTPolicy.from_pretrained(str(ckpt_dir)).to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    # -------------------------------------------------------------------------
    # [MARCA 3: EXTRAER UN INSTANTE REAL DEL BRAZO SO-101 (IMÁGENES + MOTORES)]
    # -------------------------------------------------------------------------
    print(f"\n[MARCA 3] Cargando Frame #{args.frame} del Episodio #{args.episode} ({args.repo_id})...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.local_dir,
        episodes=[args.episode],
        video_backend="pyav",
    )
    sample = dataset[args.frame]

    raw_state = sample["observation.state"].numpy()
    gt_action = sample["action"].numpy()
    cam_keys = [k for k in sample.keys() if k.startswith("observation.images.")]

    for cam_k in cam_keys:
        print(f"          • Tensor {cam_k:<24}: shape={tuple(sample[cam_k].shape)}")
    print(f"          • Estado actual 6 motores (°)  : {np.round(raw_state, 2)}")
    print(f"          • Acción humana real (°)       : {np.round(gt_action, 2)}")

    # -------------------------------------------------------------------------
    # [MARCA 4: EJECUTAR INFERENCIA EN TIEMPO REAL (PRE -> ACT -> POST)]
    # -------------------------------------------------------------------------
    batch = {"observation.state": sample["observation.state"].unsqueeze(0).to(device)}
    for cam_k in cam_keys:
        batch[cam_k] = sample[cam_k].unsqueeze(0).to(device)

    policy.reset()
    with torch.no_grad():
        t0 = time.perf_counter()
        norm_batch = preprocessor(batch)
        norm_action = extract_action_tensor(policy.select_action(norm_batch))
        phys_action_tensor = postprocessor(norm_action)
        if device.type == "mps":
            torch.mps.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

    pred_action = phys_action_tensor.squeeze(0).cpu().numpy()
    abs_err = np.abs(pred_action - gt_action)

    print(f"\n[MARCA 4] Resultado de la Inferencia en '{device.type}' (Tiempo: {infer_ms:.2f} ms):")
    print(f"          • Predicción de tu modelo ACT  : {np.round(pred_action, 2)} grados")
    print(f"          • Acción real del humano (GT)  : {np.round(gt_action, 2)} grados")
    print(f"          • Error absoluto por motor (°) : {np.round(abs_err, 2)} (Media: {np.mean(abs_err):.2f}°)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
