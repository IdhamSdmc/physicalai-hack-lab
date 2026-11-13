#!/usr/bin/env python3
"""
================================================================================
SPRINT 1 — AUDITOR DE CALIDAD DE DATASETS LEROBOT (PARQUET)
================================================================================
[¿PARA QUÉ SIRVE ESTE SCRIPT?]
En Machine Learning clásico limpias CSVs o tablas SQL antes de entrenar.
En Robótica por Imitación (Imitation Learning), tus datos son demostraciones humanas
grabadas a 30 FPS (30 filas por segundo). Si metes episodios con "basura" (cámaras
que dieron un tirón por USB, o un humano que se quedó pensando 4 segundos sin mover
el brazo), la red neuronal (ACT) aprende a quedarse congelada o a temblar.

[¿CÓMO ESTÁ CONSTRUIDO UN LEROBOT DATASET v3.0?]
Un dataset en LeRobot NO es una carpeta gigante de imágenes sueltas. Se divide en 3 partes:
  1. `meta/info.json`     -> Esquema del robot: cuántos motores tiene, FPS (ej. 30 Hz) y cámaras.
  2. `data/**/*.parquet`  -> Telemetría pura en Apache Parquet (¡tu terreno como Data Engineer!):
                             - `episode_index`: ID de la grabación (0, 1, 2... hasta ~50).
                             - `timestamp`: Segundo exacto desde que empezó el episodio (0.000, 0.033...).
                             - `observation.state`: Vector de 6 números con la posición REAL actual
                               de los 6 servomotores del brazo SO-101 (leída de los sensores).
                             - `action`: Vector de 6 números con la posición OBJETIVO que el humano
                               ordenó con el brazo maestro (leader).
  3. `videos/**/*.mp4`    -> Vídeos comprimidos de las cámaras sincronizados por timestamp.
================================================================================
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def analyze_dataset(repo_id: str, local_dir: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"[ERROR] Instala primero huggingface_hub y pyarrow: {exc}")
        sys.exit(1)

    print("=" * 78)
    print(f"  PASO 1: AUDITORÍA ANATÓMICA Y DE CALIDAD DE DATASET: {repo_id}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [MARCA 1: DESCARGA SELECTIVA DE METADATOS Y PARQUET]
    # ¿Por qué?: En el hackathon o practicando, un dataset de vídeo pesa varios GBs.
    # Usando `allow_patterns` bajamos SOLO los JSON y Parquet (~2 MB) para auditar
    # la telemetría de los motores en 3 segundos sin esperar a bajar los MP4.
    # -------------------------------------------------------------------------
    if local_dir and Path(local_dir).exists():
        root = Path(local_dir)
    else:
        print("\n[MARCA 1] Descargando solo esquema (JSON) y telemetría (Parquet) desde HF Hub...")
        print("          (Truco: filtramos los .mp4 para auditar en segundos sin gastar ancho de banda)")
        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/*.json", "meta/*.jsonl", "data/**/*.parquet", "*.parquet"],
        )
        root = Path(downloaded)

    # -------------------------------------------------------------------------
    # [MARCA 2: LECTURA DE META/INFO.JSON (EL CONTRATO DEL ROBOT)]
    # ¿Por qué?: Aquí LeRobot guarda a qué frecuencia (FPS) se grabó el robot y
    # qué dimensiones tienen los vectores de estado y acción.
    # -------------------------------------------------------------------------
    info_path = root / "meta" / "info.json"
    nominal_fps = 30.0
    if info_path.exists():
        info = json.loads(info_path.read_text())
        nominal_fps = float(info.get("fps", 30.0))
        features = info.get("features", {})
        state_shape = features.get("observation.state", {}).get("shape", "?")
        action_shape = features.get("action", {}).get("shape", "?")
        cams = [k for k in features.keys() if k.startswith("observation.images")]
        print(f"\n[MARCA 2] Esquema del Robot leído de meta/info.json:")
        print(f"          • Tipo de Robot       : {info.get('robot_type', '?')} (v{info.get('codebase_version', '?')})")
        print(f"          • Frecuencia (FPS)    : {nominal_fps} Hz (1 frame cada {1000.0/nominal_fps:.1f} ms)")
        print(f"          • Motores (State/Act) : {state_shape} estados -> {action_shape} acciones (6 grados de libertad en SO-101)")
        print(f"          • Cámaras registradas : {cams}")

    # -------------------------------------------------------------------------
    # [MARCA 3: UNIFICACIÓN DE CHUNKS PARQUET CON PYARROW]
    # ¿Cómo se hace?: LeRobot v3 guarda los episodios en bloques (`data/chunk-000/...`).
    # Los concatenamos en memoria en una sola tabla columnar ultra-rápida.
    # -------------------------------------------------------------------------
    parquet_files = sorted(root.glob("data/**/*.parquet")) or sorted(root.glob("**/*.parquet"))
    if not parquet_files:
        print(f"[ERROR] No se encontraron archivos .parquet en {root}")
        sys.exit(1)

    tables = [pq.read_table(p) for p in parquet_files]
    full_table = pa.concat_tables(tables)
    cols = full_table.column_names

    print(f"\n[MARCA 3] Tabla Parquet cargada en memoria:")
    print(f"          • Archivos Parquet    : {len(parquet_files)}")
    print(f"          • Columnas de la tabla: {cols}")
    print(f"          • Total de filas      : {full_table.num_rows:,} frames (instantes de tiempo)")

    ep_col = "episode_index" if "episode_index" in cols else None
    ts_col = "timestamp" if "timestamp" in cols else None
    state_col = "observation.state" if "observation.state" in cols else None
    action_col = "action" if "action" in cols else None

    if not ep_col or not ts_col:
        print("[!] El dataset no contiene 'episode_index' o 'timestamp' estándar.")
        return

    episodes = full_table[ep_col].to_numpy()
    timestamps = full_table[ts_col].to_numpy()
    states = np.stack(full_table[state_col].to_numpy()) if state_col else None
    actions = np.stack(full_table[action_col].to_numpy()) if action_col else None

    # Mostramos una fila real para que veas qué aspecto tienen los tensores del brazo
    if states is not None and actions is not None:
        print(f"\n[EJEMPLO DE FILA REAL (Frame #0 del Episodio #0)]:")
        print(f"          • timestamp           : {timestamps[0]:.4f} s")
        print(f"          • observation.state   : {np.round(states[0], 2)} (Posición actual de los 6 servos)")
        print(f"          • action              : {np.round(actions[0], 2)} (Posición destino enviada al brazo)")
        print(f"          • Error de seguimiento: {np.round(actions[0] - states[0], 2)} (Diferencia que mueve el motor)")

    # -------------------------------------------------------------------------
    # [MARCA 4: LAS 4 REGLAS DE AUDITORÍA FÍSICA]
    # Aquí aplicamos ciencia de datos a la robótica para cazar episodios defectuosos:
    #   Regla A (Jitter USB): ¿El salto de tiempo entre dos frames supera 1.6x (53ms)?
    #   Regla B (Hesitation): ¿El humano dejó el brazo quieto (velocidad ~ 0) >45% del tiempo?
    #   Regla C (Jerk / Tirones): ¿La 2ª derivada de la acción tiene picos violentos?
    #   Regla D (Shortcut Learning): ¿Todos los episodios empiezan en el mismo punto exacto?
    # -------------------------------------------------------------------------
    unique_eps = np.unique(episodes)
    nominal_dt = 1.0 / nominal_fps

    flagged_episodes: list[tuple[int, list[str]]] = []
    initial_states = []
    durations = []

    for ep in unique_eps:
        mask = episodes == ep
        ep_ts = timestamps[mask]
        reasons = []

        if len(ep_ts) < 10:
            reasons.append(f"Demasiado corto ({len(ep_ts)} frames)")
        else:
            dts = np.diff(ep_ts)
            durations.append(float(ep_ts[-1] - ep_ts[0]))
            max_dt = float(np.max(dts))
            drops = int(np.sum(dts > nominal_dt * 1.6))
            if drops > 0:
                reasons.append(f"{drops} caídas de FPS (máx salto={max_dt*1000:.1f}ms vs ideal {nominal_dt*1000:.1f}ms)")

        if states is not None and np.sum(mask) >= 10:
            ep_states = states[mask]
            initial_states.append(ep_states[0])
            # Velocidad = derivada primera de la posición articular entre frames consecutivos
            vel = np.linalg.norm(np.diff(ep_states, axis=0), axis=1)
            frozen_ratio = float(np.mean(vel < 1e-3))
            if frozen_ratio > 0.45:
                reasons.append(f"Hesitation alta ({frozen_ratio*100:.0f}% del tiempo el brazo estuvo inmóvil)")

            # Jerk = derivada segunda de la acción (cambios violentos de dirección)
            if actions is not None:
                ep_actions = actions[mask]
                jerk = np.linalg.norm(np.diff(ep_actions, n=2, axis=0), axis=1)
                if float(np.percentile(jerk, 98)) > float(np.median(jerk) * 15 + 1e-2):
                    reasons.append("Tirones bruscos de mando (alto Jerk P98)")

        if reasons:
            flagged_episodes.append((int(ep), reasons))

    print("\n[MARCA 4] Resultado de la Auditoría de Calidad por Episodio:")
    print(f"          • Episodios totales   : {len(unique_eps)}")
    if durations:
        print(f"          • Duración por ep.    : media={np.mean(durations):.1f}s | "
              f"min={np.min(durations):.1f}s | max={np.max(durations):.1f}s")

    if initial_states:
        init_arr = np.stack(initial_states)
        init_std = np.std(init_arr, axis=0)
        print(f"          • Diversidad postura inicial (std por motor): {np.round(init_std, 2)}")
        if float(np.mean(init_std)) < 0.01:
            print("            [!] ALERTA: Postura inicial idéntica en todos los episodios (riesgo de memorización).")

    if not flagged_episodes:
        print("\n  ✅ [DATASET LIMPIO] 0 episodios con caídas de FPS, dudas excesivas o tirones.")
    else:
        print(f"\n  ⚠️ [ATENCIÓN] Se detectaron {len(flagged_episodes)}/{len(unique_eps)} episodios con anomalías:")
        for ep_idx, r_list in flagged_episodes[:15]:
            print(f"      - Episodio #{ep_idx:03d}: " + " | ".join(r_list))
        if len(flagged_episodes) > 15:
            print(f"      ... y {len(flagged_episodes) - 15} episodios más.")
        bad_ids = [ep_idx for ep_idx, _ in flagged_episodes[:10]]
        print("\n  💡 [ACCIÓN EN EL HACKATHON]: Para borrar episodios malos sin regrabar todo, ejecuta:")
        print(f"     lerobot-edit-dataset --repo_id {repo_id} --operation.type delete_episodes --operation.episode_indices \"{bad_ids}\"")

    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audita un LeRobotDataset leyendo sus archivos Parquet.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/svla_so101_pickplace",
        help="ID del dataset en Hugging Face Hub (ej. lerobot/svla_so101_pickplace o lerobot/pusht)",
    )
    parser.add_argument("--local-dir", type=str, default=None, help="Ruta local opcional al dataset")
    args = parser.parse_args()
    analyze_dataset(args.repo_id, args.local_dir)
