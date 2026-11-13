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
del brazo SO-101 (2 cámaras + 6 ángulos) y compara la predicción del modelo contra
lo que hizo el humano.
================================================================================
"""

import glob
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main() -> None:
    print("=" * 78)
    print("  PASO 3: INSPECCIÓN DE CHECKPOINT E INFERENCIA REAL EN EL BRAZO SO-101")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [MARCA 1: LOCALIZAR EL ÚLTIMO CHECKPOINT ENTRENADO EN DISCO]
    # -------------------------------------------------------------------------
    ckpts = sorted(glob.glob("outputs/train/act_so101_smoke_*/checkpoints/*/pretrained_model"))
    if not ckpts:
        print("[ERROR] No se encontró ningún checkpoint en outputs/train/. Ejecuta primero 03_train_local_m5.sh")
        sys.exit(1)

    ckpt_dir = Path(ckpts[-1])
    print(f"\n[MARCA 1] Checkpoint detectado : {ckpt_dir}")
    for f in sorted(ckpt_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{f.stat().st_size / 1024:.1f} KB"
        print(f"          • {f.name:<62} ({size_str})")

    # -------------------------------------------------------------------------
    # [MARCA 2: CARGAR POLÍTICA ACT + PREPROCESSOR + POSTPROCESSOR]
    # ¿Por qué es vital en LeRobot v0.6?:
    # En versiones antiguas la normalización iba dentro del modelo; en v0.6 va en
    # `PolicyProcessorPipeline` separados (`policy_preprocessor` y `policy_postprocessor`).
    # -------------------------------------------------------------------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[MARCA 2] Cargando pesos (51.5M parámetros) en dispositivo: {device}...")

    policy = ACTPolicy.from_pretrained(str(ckpt_dir)).to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt_dir),
    )

    # -------------------------------------------------------------------------
    # [MARCA 3: EXTRAER UN INSTANTE REAL DEL BRAZO SO-101 (IMÁGENES + MOTORES)]
    # -------------------------------------------------------------------------
    print("\n[MARCA 3] Cargando Frame #10 del Episodio #0 desde caché local...")
    dataset = LeRobotDataset(
        repo_id="lerobot/svla_so101_pickplace",
        episodes=[0],
        video_backend="pyav",
    )
    sample = dataset[10]

    raw_state = sample["observation.state"].numpy()
    gt_action = sample["action"].numpy()
    img_up_shape = tuple(sample["observation.images.up"].shape)
    img_side_shape = tuple(sample["observation.images.side"].shape)

    print(f"          • Tensor Cámara Cenital (up)   : shape={img_up_shape} (Canales, Alto, Ancho)")
    print(f"          • Tensor Cámara Lateral (side) : shape={img_side_shape}")
    print(f"          • Estado actual 6 motores (°)  : {np.round(raw_state, 2)}")
    print(f"          • Acción humana real (°)       : {np.round(gt_action, 2)}")

    # -------------------------------------------------------------------------
    # [MARCA 4: EJECUTAR INFERENCIA EN TIEMPO REAL (PRE -> ACT -> POST)]
    # `select_action` gestiona automáticamente la cola de Action Chunking /
    # Temporal Ensembling para devolver la acción exacta que debes enviar por USB
    # a los servos en este ciclo de 33.3 ms.
    # -------------------------------------------------------------------------
    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0).to(device),
        "observation.images.up": sample["observation.images.up"].unsqueeze(0).to(device),
        "observation.images.side": sample["observation.images.side"].unsqueeze(0).to(device),
    }

    policy.reset()
    with torch.no_grad():
        t0 = time.perf_counter()
        norm_batch = preprocessor(batch)
        norm_action = policy.select_action(norm_batch)
        phys_action_tensor = postprocessor(norm_action)
        if device.type == "mps":
            torch.mps.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

    pred_action = phys_action_tensor.squeeze(0).cpu().numpy()
    abs_err = np.abs(pred_action - gt_action)

    print(f"\n[MARCA 4] Resultado de la Inferencia en tu Apple M5 (Tiempo: {infer_ms:.2f} ms):")
    print(f"          • Predicción de tu modelo ACT  : {np.round(pred_action, 2)} grados")
    print(f"          • Acción real del humano (GT)  : {np.round(gt_action, 2)} grados")
    print(f"          • Error absoluto por motor (°) : {np.round(abs_err, 2)} (Media: {np.mean(abs_err):.2f}° en solo 100 pasos!)")
    print("=" * 78)


if __name__ == "__main__":
    main()
