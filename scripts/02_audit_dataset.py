#!/usr/bin/env python3
"""
================================================================================
SPRINT 1 — AUDITOR DE CALIDAD DE DATASETS LEROBOT (PARQUET)
================================================================================
[¿PARA QUÉ SIRVE ESTE SCRIPT?]
En Machine Learning clásico limpias CSVs o tablas SQL antes de entrenar.
En Robótica por Imitación (Imitation Learning), tus datos son demostraciones humanas
grabadas a 30 FPS (30 filas por segundo). Si metes episodios con "basura" (cámaras
que dieron un tirón por USB, un humano que se quedó parado al principio o a mitad
del episodio, o tirones bruscos), la red neuronal (ACT) aprende a quedarse congelada
o a temblar.

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
from typing import Any

import numpy as np


def _count_leading_idle_frames(vel: np.ndarray, threshold: float = 1e-3) -> int:
    """Count consecutive stationary frames from episode start."""
    count = 0
    for step_vel in vel:
        if float(step_vel) < threshold:
            count += 1
        else:
            break
    return count


def _read_tasks_metadata(root: Path, pq: Any) -> list[str]:
    """Read task strings from LeRobot v3 tasks.parquet or legacy tasks.jsonl."""
    tasks_pq = root / "meta" / "tasks.parquet"
    if tasks_pq.exists():
        table = pq.read_table(tasks_pq)
        if "task" in table.column_names:
            return [str(v) for v in table["task"].to_pylist() if str(v).strip()]
        # En algunos datasets v3, el texto de la tarea está en el índice pandas guardado en Parquet
        df = table.to_pandas()
        return [str(idx) for idx in df.index if str(idx).strip()]

    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks: list[str] = []
        for line in tasks_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("task"):
                tasks.append(str(item["task"]))
        return tasks
    return []


def audit_dataset_root(
    root: Path,
    repo_id: str,
    max_idle_start_s: float = 1.5,
    max_hesitation_ratio: float = 0.45,
) -> dict[str, Any]:
    """Audit metadata and Parquet telemetry under `root` and return structured findings."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    info: dict[str, Any] = {}
    nominal_fps = 30.0
    cams: list[str] = []
    camera_shapes: dict[str, Any] = {}
    state_shape: Any = "?"
    action_shape: Any = "?"

    if info_path.exists():
        info = json.loads(info_path.read_text())
        nominal_fps = float(info.get("fps", 30.0))
        features = info.get("features", {})
        state_shape = features.get("observation.state", {}).get("shape", "?")
        action_shape = features.get("action", {}).get("shape", "?")
        cams = [k for k in features.keys() if k.startswith("observation.images")]
        camera_shapes = {k: features[k].get("shape") for k in cams}

    tasks = _read_tasks_metadata(root, pq)

    parquet_files = sorted(root.glob("data/**/*.parquet")) or sorted(root.glob("**/*.parquet"))
    parquet_files = [p for p in parquet_files if p.parent.name != "meta"]
    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos .parquet en {root}")

    tables = [pq.read_table(p) for p in parquet_files]
    full_table = pa.concat_tables(tables)
    cols = full_table.column_names

    if "episode_index" not in cols or "timestamp" not in cols:
        raise ValueError("El dataset no contiene 'episode_index' o 'timestamp' estándar.")

    episodes = full_table["episode_index"].to_numpy()
    timestamps = full_table["timestamp"].to_numpy()
    frame_indices = full_table["frame_index"].to_numpy() if "frame_index" in cols else None
    states = np.stack(full_table["observation.state"].to_numpy()) if "observation.state" in cols else None
    actions = np.stack(full_table["action"].to_numpy()) if "action" in cols else None

    unique_eps = np.unique(episodes)
    nominal_dt = 1.0 / nominal_fps if nominal_fps > 0 else 1.0 / 30.0

    flagged_episodes: list[tuple[int, list[str]]] = []
    initial_states: list[np.ndarray] = []
    durations: list[float] = []
    leading_idle_secs: list[float] = []

    for ep in unique_eps:
        mask = episodes == ep
        ep_ts = timestamps[mask]
        ep_states = states[mask] if states is not None else None
        ep_actions = actions[mask] if actions is not None else None

        # Garantizar orden temporal dentro del episodio antes de calcular derivadas
        order_key = frame_indices[mask] if frame_indices is not None else ep_ts
        order = np.argsort(order_key, kind="stable")
        ep_ts = ep_ts[order]
        if ep_states is not None:
            ep_states = ep_states[order]
        if ep_actions is not None:
            ep_actions = ep_actions[order]

        reasons: list[str] = []
        if len(ep_ts) < 10:
            reasons.append(f"Demasiado corto ({len(ep_ts)} frames)")
        else:
            dts = np.diff(ep_ts)
            durations.append(float(ep_ts[-1] - ep_ts[0]))
            max_dt = float(np.max(dts))
            drops = int(np.sum(dts > nominal_dt * 1.6))
            if drops > 0:
                reasons.append(
                    f"{drops} caídas de FPS (máx salto={max_dt * 1000:.1f}ms vs ideal {nominal_dt * 1000:.1f}ms)"
                )

        if ep_states is not None and len(ep_states) >= 10:
            initial_states.append(ep_states[0])
            vel = np.linalg.norm(np.diff(ep_states, axis=0), axis=1)
            frozen_ratio = float(np.mean(vel < 1e-3))
            if frozen_ratio > max_hesitation_ratio:
                reasons.append(f"Hesitation alta ({frozen_ratio * 100:.0f}% del tiempo el brazo estuvo inmóvil)")

            idle_frames = _count_leading_idle_frames(vel)
            idle_s = idle_frames * nominal_dt
            leading_idle_secs.append(idle_s)
            if idle_s > max_idle_start_s:
                reasons.append(f"Tiempo muerto inicial ({idle_s:.1f}s quieto al arrancar; recorta o mueve antes)")

            if ep_actions is not None:
                jerk = np.linalg.norm(np.diff(ep_actions, n=2, axis=0), axis=1)
                if float(np.percentile(jerk, 98)) > float(np.median(jerk) * 15 + 1e-2):
                    reasons.append("Tirones bruscos de mando (alto Jerk P98)")

        if reasons:
            flagged_episodes.append((int(ep), reasons))

    init_std: list[float] = []
    init_sensor_freeze = False
    if initial_states:
        init_arr = np.stack(initial_states)
        std_arr = np.std(init_arr, axis=0)
        init_std = [round(float(x), 4) for x in std_arr]
        init_sensor_freeze = len(initial_states) > 1 and float(np.max(std_arr)) < 1e-4

    schema_warnings: list[str] = []
    if info:
        if abs(nominal_fps - 30.0) > 1.0:
            schema_warnings.append(f"FPS={nominal_fps} (la guía SO-101 recomienda 30 FPS).")
        if not cams:
            schema_warnings.append("No se detectaron cámaras `observation.images.*` en meta/info.json.")
        if not tasks:
            schema_warnings.append("No se encontró descripción de tarea en meta/tasks.")

    total_eps = int(len(unique_eps))
    flagged_ids = [ep_idx for ep_idx, _ in flagged_episodes]
    clean_eps = total_eps - len(flagged_ids)

    return {
        "repo_id": repo_id,
        "root": str(root),
        "robot_type": info.get("robot_type", "?"),
        "codebase_version": info.get("codebase_version", "?"),
        "nominal_fps": nominal_fps,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "cameras": cams,
        "camera_shapes": camera_shapes,
        "tasks": tasks,
        "parquet_files": len(parquet_files),
        "columns": cols,
        "total_frames": int(full_table.num_rows),
        "total_episodes": total_eps,
        "clean_episodes": clean_eps,
        "flagged_count": len(flagged_ids),
        "discard_rate_pct": round((len(flagged_ids) / total_eps * 100.0) if total_eps else 0.0, 1),
        "flagged_episodes": [{"episode": ep_idx, "reasons": r_list} for ep_idx, r_list in flagged_episodes],
        "flagged_ids": flagged_ids,
        "duration_mean_s": round(float(np.mean(durations)), 2) if durations else 0.0,
        "duration_min_s": round(float(np.min(durations)), 2) if durations else 0.0,
        "duration_max_s": round(float(np.max(durations)), 2) if durations else 0.0,
        "leading_idle_mean_s": round(float(np.mean(leading_idle_secs)), 2) if leading_idle_secs else 0.0,
        "initial_state_std": init_std,
        "initial_sensor_freeze": init_sensor_freeze,
        "schema_warnings": schema_warnings,
        "sample_row": (
            {
                "timestamp": float(timestamps[0]),
                "observation.state": np.round(states[0], 2).tolist(),
                "action": np.round(actions[0], 2).tolist(),
                "tracking_error": np.round(actions[0] - states[0], 2).tolist(),
            }
            if states is not None and actions is not None and len(timestamps) > 0
            else None
        ),
    }


def analyze_dataset(
    repo_id: str,
    local_dir: str | None = None,
    max_idle_start_s: float = 1.5,
    max_hesitation_ratio: float = 0.45,
    json_out: str | None = None,
) -> dict[str, Any]:
    """Download metadata/Parquet if needed, run quality audit, print report, and return summary."""
    try:
        from huggingface_hub import snapshot_download
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        print(f"[ERROR] Instala primero huggingface_hub y pyarrow: {exc}")
        sys.exit(1)

    print("=" * 78)
    print(f"  PASO 1: AUDITORÍA ANATÓMICA Y DE CALIDAD DE DATASET: {repo_id}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [MARCA 1: DESCARGA SELECTIVA DE METADATOS Y PARQUET]
    # -------------------------------------------------------------------------
    if local_dir and Path(local_dir).exists():
        root = Path(local_dir)
    else:
        print("\n[MARCA 1] Descargando solo esquema (JSON) y telemetría (Parquet) desde HF Hub...")
        print("          (Truco: filtramos los .mp4 para auditar en segundos sin gastar ancho de banda)")
        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/*.json", "meta/*.jsonl", "meta/*.parquet", "data/**/*.parquet", "*.parquet"],
        )
        root = Path(downloaded)

    try:
        report = audit_dataset_root(
            root,
            repo_id=repo_id,
            max_idle_start_s=max_idle_start_s,
            max_hesitation_ratio=max_hesitation_ratio,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # [MARCA 2: LECTURA DE META/INFO.JSON (EL CONTRATO DEL ROBOT)]
    # -------------------------------------------------------------------------
    print("\n[MARCA 2] Esquema del Robot leído de meta/info.json:")
    ver_label = str(report["codebase_version"]).lstrip("v")
    print(f"          • Tipo de Robot       : {report['robot_type']} (v{ver_label})")
    print(f"          • Frecuencia (FPS)    : {report['nominal_fps']} Hz (1 frame cada {1000.0 / report['nominal_fps']:.1f} ms)")
    print(f"          • Motores (State/Act) : {report['state_shape']} estados -> {report['action_shape']} acciones")
    print(f"          • Cámaras registradas : {report['camera_shapes'] or report['cameras']}")
    if report["tasks"]:
        print(f"          • Tarea(s)            : {report['tasks']}")
    for warn in report["schema_warnings"]:
        print(f"          [!] AVISO DE ESQUEMA  : {warn}")

    # -------------------------------------------------------------------------
    # [MARCA 3: UNIFICACIÓN DE CHUNKS PARQUET CON PYARROW]
    # -------------------------------------------------------------------------
    print("\n[MARCA 3] Tabla Parquet cargada en memoria:")
    print(f"          • Archivos Parquet    : {report['parquet_files']}")
    print(f"          • Columnas de la tabla: {report['columns']}")
    print(f"          • Total de filas      : {report['total_frames']:,} frames (instantes de tiempo)")

    if report["sample_row"]:
        row = report["sample_row"]
        print("\n[EJEMPLO DE FILA REAL (Frame #0 del Episodio #0)]:")
        print(f"          • timestamp           : {row['timestamp']:.4f} s")
        print(f"          • observation.state   : {row['observation.state']} (Posición actual de los servos)")
        print(f"          • action              : {row['action']} (Posición destino enviada al brazo)")
        print(f"          • Error de seguimiento: {row['tracking_error']}")

    # -------------------------------------------------------------------------
    # [MARCA 4: RESULTADO DE AUDITORÍA + RESUMEN PARA EL JURADO DE BILBAO]
    # -------------------------------------------------------------------------
    total_eps = report["total_episodes"]
    clean_eps = report["clean_episodes"]
    print("\n[MARCA 4] Resultado de la Auditoría de Calidad por Episodio:")
    print(f"          • Episodios totales   : {total_eps} (Limpios: {clean_eps} | Marcados: {report['flagged_count']} / {report['discard_rate_pct']}%)")
    if total_eps:
        print(
            f"          • Duración por ep.    : media={report['duration_mean_s']:.1f}s | "
            f"min={report['duration_min_s']:.1f}s | max={report['duration_max_s']:.1f}s"
        )
        print(f"          • Tiempo muerto inicio: media={report['leading_idle_mean_s']:.2f}s (umbral aviso > {max_idle_start_s:.1f}s)")

    if report["initial_state_std"]:
        print(f"          • Postura inicial std : {report['initial_state_std']} (la guía recomienda empezar desde Home consistente)")
        if report["initial_sensor_freeze"]:
            print("            [!] ALERTA: Lectura inicial idéntica bit-a-bit (posible sensor congelado o duplicado).")

    if clean_eps < 20:
        print("          • Nivel guía Seeed/SO-101: <20 limpios (solo smoke test; graba ~20 para fijo / ~50 aleatorio).")
    elif clean_eps < 50:
        print("          • Nivel guía Seeed/SO-101: Inicial (~20-49 limpios, recomendado 10.000-15.000 pasos ACT).")
    else:
        print("          • Nivel guía Seeed/SO-101: Intermedio/Avanzado (>=50 limpios, recomendado 30.000+ pasos ACT).")

    if not report["flagged_episodes"]:
        print("\n  ✅ [DATASET LIMPIO] 0 episodios con caídas de FPS, dudas excesivas, tiempo muerto inicial o tirones.")
    else:
        print(f"\n  ⚠️ [ATENCIÓN] Se detectaron {report['flagged_count']}/{total_eps} episodios con anomalías:")
        for item in report["flagged_episodes"][:15]:
            print(f"      - Episodio #{item['episode']:03d}: " + " | ".join(item["reasons"]))
        if len(report["flagged_episodes"]) > 15:
            print(f"      ... y {len(report['flagged_episodes']) - 15} episodios más.")
        bad_ids = report["flagged_ids"]
        root_flag = f" --root {local_dir}" if local_dir else ""
        print("\n  💡 [ACCIÓN EN EL HACKATHON]: Para borrar episodios malos sin regrabar todo, ejecuta:")
        print(f"     lerobot-edit-dataset --repo_id {repo_id}{root_flag} --operation.type delete_episodes --operation.episode_indices \"{bad_ids}\"")
        print(f"     (O fíltralos al vuelo al entrenar con: --dataset.exclude_episodes=\"{bad_ids}\")")

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n  📄 [JURADO] Reporte JSON guardado en: {out_path}")

    print("=" * 78)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audita un LeRobotDataset leyendo sus archivos Parquet.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/svla_so101_pickplace",
        help="ID del dataset en Hugging Face Hub (ej. lerobot/svla_so101_pickplace o lerobot/pusht)",
    )
    parser.add_argument("--local-dir", type=str, default=None, help="Ruta local opcional al dataset")
    parser.add_argument("--max-idle-start-s", type=float, default=1.5, help="Máximo tiempo muerto al inicio del episodio (s)")
    parser.add_argument("--max-hesitation-ratio", type=float, default=0.45, help="Ratio máximo de frames quietos por episodio")
    parser.add_argument("--json-out", type=str, default=None, help="Ruta opcional para exportar el resumen JSON del auditor")
    args = parser.parse_args()
    analyze_dataset(
        args.repo_id,
        args.local_dir,
        max_idle_start_s=args.max_idle_start_s,
        max_hesitation_ratio=args.max_hesitation_ratio,
        json_out=args.json_out,
    )
