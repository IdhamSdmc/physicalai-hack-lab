#!/usr/bin/env python3
"""
================================================================================
PASO 4.1 — VISUALIZACIÓN DE EPISODIOS (RERUN) Y DATA AUGMENTATION DE CÁMARAS
================================================================================
[¿PARA QUÉ SIRVE ESTE PASO EN EL HACKATHON?]
1. Inspección Visual de Episodios (`lerobot-dataset-viz`):
   Antes de entrenar 30 minutos, debes abrir al menos 2 episodios grabados para
   comprobar que:
   - Las dos cámaras (`up` y `side`) no están intercambiadas ni desenfocadas.
   - El objeto objetivo (cubo/pieza) entra siempre dentro del campo visual (FOV)
     de AMBAS cámaras en el momento del agarre.
2. Previsualización de Data Augmentation (`lerobot-imgtransform-viz`):
   En un pabellón como La Perrera (Bilbao), la luz cambia de la mañana a la noche
   y las sombras de la gente alrededor de la mesa confunden a la red neuronal.
   Activar `--dataset.image_transforms.enable=true` aplica variaciones aleatorias
   de brillo, contraste, saturación, nitidez y matiz (Hue) a cada frame durante
   el entrenamiento para que el modelo ignore la iluminación y se fije solo en la
   geometría del objeto y la pinza.
================================================================================
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage


def main() -> None:
    print("=" * 78)
    print("  PASO 4.1: VISUALIZACIÓN DE EPISODIO #0 (RERUN) Y DATA AUGMENTATION")
    print("=" * 78)

    venv_bin = Path(sys.executable).parent
    viz_dir = Path("outputs/viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # [MARCA 1: EXPORTAR EPISODIO #0 A FORMATO RERUN (.rrd)]
    # ¿Cómo funciona?: `lerobot-dataset-viz` empaqueta los vídeos de las cámaras
    # y las 6 curvas de los servomotores en una línea de tiempo interactiva.
    # Con `--save 1` guardamos el archivo `.rrd` en disco (y sin `--save 1` se
    # abre directamente la ventana interactiva de Rerun en tu Mac).
    # -------------------------------------------------------------------------
    print("\n[MARCA 1] Generando grabación interactiva Rerun (.rrd) del Episodio #0...")
    cmd_viz = [
        str(venv_bin / "lerobot-dataset-viz"),
        "--repo-id", "lerobot/svla_so101_pickplace",
        "--episode-index", "0",
        "--save", "1",
        "--output-dir", str(viz_dir),
    ]
    subprocess.run(cmd_viz, check=True)
    rrd_files = list(viz_dir.glob("*.rrd"))
    if rrd_files:
        print(f"          • Archivo Rerun generado : {rrd_files[0]} ({rrd_files[0].stat().st_size / (1024*1024):.1f} MB)")
        print(f"          • Para abrir el visor 3D interactivo en tu Mac ejecuta:")
        print(f"            .venv/bin/rerun {rrd_files[0]}")

    # -------------------------------------------------------------------------
    # [MARCA 2: GENERAR VARIACIONES DE DATA AUGMENTATION (lerobot-imgtransform-viz)]
    # ¿Por qué?: Te permite verificar que las transformaciones de color/brillo no
    # sean tan extremas que oculten el color de la pieza que el robot debe coger.
    # -------------------------------------------------------------------------
    print("\n[MARCA 2] Ejecutando lerobot-imgtransform-viz sobre el Episodio #0...")
    cmd_tf = [
        str(venv_bin / "lerobot-imgtransform-viz"),
        "--repo_id=lerobot/svla_so101_pickplace",
        "--episodes=[0]",
        "--video_backend=pyav",
        "--image_transforms.enable=true",
    ]
    subprocess.run(cmd_tf, check=True)

    tf_root = Path("outputs/image_transforms/svla_so101_pickplace")
    print("          • Transformaciones generadas en disco:")
    for sub in sorted(tf_root.iterdir()):
        if sub.is_dir():
            pngs = list(sub.glob("*.png"))
            print(f"            - {sub.name:<15}: {len(pngs)} ejemplos ({', '.join(p.name for p in sorted(pngs)[:4])}...)")

    # -------------------------------------------------------------------------
    # [MARCA 3: CONSTRUIR LÁMINA VISUAL DE LA TRAYECTORIA Y CÁMARAS]
    # Extraemos 5 instantes clave del Episodio #0 (inicio, aproximación, agarre,
    # traslado y depósito) para analizar qué ven las cámaras `up` y `side` y cómo
    # cambian los grados del servo #6 (pinza).
    # -------------------------------------------------------------------------
    print("\n[MARCA 3] Analizando 5 fases clave del Episodio #0 (Pick & Place)...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id="lerobot/svla_so101_pickplace",
        episodes=[0],
        video_backend="pyav",
    )
    total_frames = len(ds)
    key_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    to_pil = ToPILImage()

    thumb_w, thumb_h = 320, 240
    cols = 5
    # Fila 1: Cámara UP (5 frames), Fila 2: Cámara SIDE (5 frames), Fila 3: 5 Data Augmentations
    board = Image.new("RGB", (cols * thumb_w, 3 * (thumb_h + 44)), color=(18, 20, 26))
    draw = ImageDraw.Draw(board)

    print(f"          Total de frames en Episodio #0: {total_frames} ({total_frames/30.0:.1f} segundos)")
    for col_idx, frame_idx in enumerate(key_indices):
        item = ds[int(frame_idx)]
        ts = float(item["timestamp"].item())
        state = item["observation.state"].numpy()
        gripper_deg = float(state[5])

        img_up = to_pil(item["observation.images.up"]).resize((thumb_w, thumb_h))
        img_side = to_pil(item["observation.images.side"]).resize((thumb_w, thumb_h))

        x0 = col_idx * thumb_w
        board.paste(img_up, (x0, 36))
        draw.text(
            (x0 + 8, 8),
            f"UP | Frame {frame_idx} (t={ts:.1f}s) | Pinza:{gripper_deg:.1f} deg",
            fill=(230, 240, 255),
        )

        y_side = thumb_h + 44
        board.paste(img_side, (x0, y_side + 36))
        draw.text(
            (x0 + 8, y_side + 8),
            f"SIDE | Motores[1..3]: [{state[0]:.0f}, {state[1]:.0f}, {state[2]:.0f}] deg",
            fill=(200, 255, 210),
        )

        print(
            f"          • Frame #{frame_idx:03d} (t={ts:4.1f}s) -> "
            f"Cintura:{state[0]:6.1f}° | Hombro:{state[1]:6.1f}° | Codo:{state[2]:6.1f}° | "
            f"Pinza(Servo#6):{gripper_deg:5.1f}°"
        )

    # Fila 3: Pegar 5 ejemplos de Data Augmentation generados por lerobot-imgtransform-viz
    aug_samples = [
        ("Brillo Mínimo (Noche)", tf_root / "brightness" / "min.png"),
        ("Brillo Máximo (Foco)", tf_root / "brightness" / "max.png"),
        ("Contraste Máximo", tf_root / "contrast" / "max.png"),
        ("Cambio de Matiz (Hue)", tf_root / "hue" / "max.png"),
        ("Combinado (Entrenamiento)", tf_root / "all" / "1.png"),
    ]
    y_aug = 2 * (thumb_h + 44)
    for col_idx, (label, path) in enumerate(aug_samples):
        if path.exists():
            aug_img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
            x0 = col_idx * thumb_w
            board.paste(aug_img, (x0, y_aug + 36))
            draw.text((x0 + 8, y_aug + 8), f"AUGMENT: {label}", fill=(255, 220, 140))

    summary_path = viz_dir / "so101_episode0_and_augmentations.png"
    board.save(summary_path)
    print(f"\n[MARCA 4] Panel visual guardado en: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
