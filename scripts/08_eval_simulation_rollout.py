#!/usr/bin/env python3
"""
================================================================================
PASO 4.3 — ENTRENAMIENTO Y EVALUACIÓN EN SIMULADOR (PUSHT) CON DIFFUSION POLICY
================================================================================
[¿PARA QUÉ SIRVE ESTE PASO?]
Hasta ahora vimos `ACT` (Action Chunking with Transformers). La segunda gran
arquitectura que te recomendó Gemini dominar es **Diffusion Policy**:
  - ¿Cómo funciona?: En lugar de predecir la trayectoria de un golpe como ACT,
    parte de ruido gaussiano puro y aplica pasos de des-ruido (igual que Stable
    Diffusion genera imágenes, pero aquí genera trayectorias de motor $[T \times D]$).
  - ¿Cuándo usar Diffusion frente a ACT en el Hackathon?:
    * Usa **ACT** por defecto (es 3x más rápida entrenando e infiriendo, ideal para
      agarrar y colocar objetos).
    * Usa **Diffusion Policy** si la tarea es de **empuje continuo / contacto rico**
      (como empujar un bloque en forma de T hasta encajarlo en una silueta, limpiar
      una mesa o arrastrar objetos donde hay múltiples caminos igual de válidos).

En este script:
  1. Entrenamos una **Diffusion Policy** nativa en LeRobot v0.6 sobre `lerobot/pusht`
     en tu Apple M5 (`mps`).
  2. Ejecutamos `lerobot-eval` sobre el checkpoint generado para lanzar el simulador
     físico **PushT** y grabar los episodios de evaluación en MP4.
  3. Analizamos el archivo `eval_info.json` (`pc_success`, `avg_max_reward`) y
     extraemos la tira visual de los frames del simulador.
================================================================================
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    print("=" * 78)
    print("  PASO 4.3: DIFFUSION POLICY + EVALUACIÓN EN SIMULADOR (PUSHT) EN APPLE M5")
    print("=" * 78)

    venv_bin = Path(sys.executable).parent
    train_dir = Path("outputs/train/diffusion_pusht_smoke")
    ckpt_dir = train_dir / "checkpoints" / "last" / "pretrained_model"
    out_dir = Path("outputs/eval/diffusion_pusht_m5")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # -------------------------------------------------------------------------
    # [MARCA 1: ENTRENAR UN CHECKPOINT NATIVO DE DIFFUSION POLICY EN TU M5]
    # ¿Por qué entrenamos nuestro propio checkpoint v0.6?:
    # Los modelos antiguos del Hub (`lerobot/diffusion_pusht`) usan el formato v0.4
    # sin `policy_preprocessor.json`. Al entrenar con `lerobot-train --policy.type=diffusion`,
    # generamos un checkpoint Diffusion 100% nativo de v0.6 en tu GPU Metal (`mps`).
    # -------------------------------------------------------------------------
    if not (ckpt_dir / "model.safetensors").exists():
        if train_dir.exists():
            shutil.rmtree(train_dir)
        print("\n[MARCA 1] Entrenando Diffusion Policy nativa (40 pasos) sobre lerobot/pusht en M5...")
        cmd_train = [
            str(venv_bin / "lerobot-train"),
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0, 1, 2, 3, 4]",
            "--dataset.video_backend=pyav",
            "--policy.type=diffusion",
            "--policy.device=mps",
            "--policy.push_to_hub=false",
            "--batch_size=8",
            "--num_workers=0",
            "--steps=40",
            "--log_freq=20",
            "--env_eval_freq=0",
            "--save_checkpoint=true",
            "--save_freq=40",
            f"--output_dir={train_dir}",
            "--trackio.enable=false",
        ]
        subprocess.run(cmd_train, check=True)
    else:
        print(f"\n[MARCA 1] Checkpoint Diffusion Policy v0.6 ya disponible en: {ckpt_dir}")

    # -------------------------------------------------------------------------
    # [MARCA 2: EJECUTAR LEROBOT-EVAL EN EL SIMULADOR FÍSICO PUSHT]
    # `lerobot-eval` levanta el motor de física 2D (PyMunk), le pasa al modelo las
    # observaciones visuales (96x96) + posición del agente (X, Y) y ejecuta los
    # comandos de movimiento devueltos por la Diffusion Policy.
    # -------------------------------------------------------------------------
    print("\n[MARCA 2] Lanzando lerobot-eval con tu Diffusion Policy en el simulador PushT...")
    cmd_eval = [
        str(venv_bin / "lerobot-eval"),
        f"--policy.path={ckpt_dir}",
        "--env.type=pusht",
        "--eval.batch_size=1",
        "--eval.n_episodes=2",
        "--eval.use_async_envs=false",
        "--policy.use_amp=false",
        "--policy.device=mps",
        f"--output_dir={out_dir}",
    ]
    subprocess.run(cmd_eval, check=True)

    # -------------------------------------------------------------------------
    # [MARCA 3: ANÁLISIS DE EVAL_INFO.JSON (MÉTRICAS DE ROLLOUT)]
    # -------------------------------------------------------------------------
    eval_json_path = out_dir / "eval_info.json"
    if eval_json_path.exists():
        info = json.loads(eval_json_path.read_text())
        overall = info.get("overall", info)
        print("\n[MARCA 3] Métricas oficiales devueltas por el simulador (eval_info.json):")
        print(f"          • Tasa de Éxito (pc_success)       : {overall.get('pc_success', '?')}%")
        print(f"          • Recompensa Máxima Media          : {overall.get('avg_max_reward', 0.0):.3f} (1.0 = encaje perfecto)")
        print(f"          • Recompensa Acumulada Media       : {overall.get('avg_sum_reward', 0.0):.1f}")
        print(f"          • Tiempo medio por episodio eval   : {overall.get('eval_ep_s', 0.0):.2f} s")

        per_ep = overall.get("per_episode", info.get("per_episode", []))
        if per_ep:
            print("          • Detalle por episodio simulado:")
            for ep in per_ep:
                print(
                    f"            - Ep #{ep.get('episode_ix', '?')}: "
                    f"success={ep.get('success')} | max_reward={ep.get('max_reward', 0):.3f} | "
                    f"sum_reward={ep.get('sum_reward', 0):.1f}"
                )

    # -------------------------------------------------------------------------
    # [MARCA 4: EXTRAER SECUENCIA VISUAL DEL VÍDEO DE ROLLOUT GENERADO]
    # -------------------------------------------------------------------------
    mp4_files = sorted(out_dir.glob("videos/**/*.mp4"))
    if mp4_files:
        video_path = mp4_files[0]
        print(f"\n[MARCA 4] Vídeo de Rollout generado por LeRobot: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if frames:
            idxs = np.linspace(0, len(frames) - 1, 6, dtype=int)
            thumb_s = 256
            strip = Image.new("RGB", (6 * thumb_s, thumb_s + 40), color=(20, 22, 28))
            draw = ImageDraw.Draw(strip)
            for col_i, f_idx in enumerate(idxs):
                im = Image.fromarray(frames[int(f_idx)]).resize((thumb_s, thumb_s))
                x0 = col_i * thumb_s
                strip.paste(im, (x0, 36))
                draw.text((x0 + 8, 8), f"PushT Rollout | Step {f_idx}/{len(frames)-1}", fill=(220, 255, 220))

            viz_path = Path("outputs/viz/pusht_rollout_summary.png")
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            strip.save(viz_path)
            print(f"          • Secuencia visual del Rollout guardada en: {viz_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
