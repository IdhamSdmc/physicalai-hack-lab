#!/usr/bin/env python3
"""
================================================================================
PASO 4.2 — DIAGNÓSTICO DE PUERTOS SERIE Y STRESS TEST DE CÁMARAS USB (macOS/Linux)
================================================================================
[¿PARA QUÉ SIRVE ESTE SCRIPT EN EL DÍA CERO DEL HACKATHON?]
El 90% de los problemas de las primeras 3 horas en un hackathon de LeRobot son físicos:
  1. ¿Qué cable USB corresponde al brazo Leader (maestro) y cuál al Follower (esclavo)?
  2. ¿Qué índice de cámara (0, 1, 2) es la cámara de la pinza (`wrist`), cuál es
     la cámara cenital (`top`) y cuál es la webcam integrada del MacBook?
  3. ¿Pueden las cámaras mantener 30 FPS clavados (33.3 ms por frame) sin saturar
     el bus USB ni cambiar el brillo solas (Auto-Exposure)?
================================================================================
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROBOT_SERIAL_TOKENS = ("usbmodem", "usbserial", "ttyACM", "ttyUSB")


def is_robot_serial_port(device: str) -> bool:
    """Return True if a serial device path matches SO-101 USB adapters on macOS or Linux."""
    return any(token in device for token in ROBOT_SERIAL_TOKENS)


def scan_serial_ports() -> list[dict[str, str]]:
    """Scan connected serial ports and flag likely SO-101 controller boards."""
    print("\n[MARCA 1] Escaneando Puertos Serie USB (donde se conectan los brazos SO-101)...")
    print("          • En Linux/WSL se llaman : /dev/ttyACM* o /dev/ttyUSB*")
    print("          • En macOS se llaman     : /dev/tty.usbmodem* (ej. /dev/tty.usbmodem585A0076841)")
    detected: list[dict[str, str]] = []
    try:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        if not ports:
            print("          • Puertos detectados : Ninguno conectado ahora mismo.")
        else:
            for p in ports:
                is_robot = is_robot_serial_port(p.device)
                tag = " <-- [PLACA SERVOS SO-101]" if is_robot else " (Puerto interno)"
                print(f"          • {p.device:<32} | desc='{p.description}'{tag}")
                detected.append({"device": p.device, "description": str(p.description), "is_robot": str(is_robot)})
    except ImportError:
        print("          [!] pyserial no disponible.")

    print("          💡 [TRUCO EN BILBAO]: Ejecuta 'lerobot-find-port'. Te pedirá desconectar")
    print("             el cable USB de un brazo y te dirá por diferencia qué puerto exacto era.")
    return detected


def compute_stream_metrics(
    timestamps: list[float],
    brightness_means: dict[int, list[float]],
    dropped_reads: int,
    target_fps: int,
) -> dict[str, Any]:
    """Compute FPS, P95/P99 frame interval jitter, and auto-exposure brightness drift."""
    if len(timestamps) < 2:
        raise ValueError("Se necesitan al menos 2 frames válidos para calcular métricas.")

    dts_ms = np.diff(timestamps) * 1000.0
    elapsed = timestamps[-1] - timestamps[0]
    real_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    mean_dt_ms = float(np.mean(dts_ms))
    p95_dt_ms = float(np.percentile(dts_ms, 95))
    p99_dt_ms = float(np.percentile(dts_ms, 99))

    drift_by_cam = {
        idx: (float(np.max(vals) - np.min(vals)) if vals else 0.0)
        for idx, vals in brightness_means.items()
    }
    return {
        "frames": len(timestamps),
        "dropped_reads": dropped_reads,
        "target_fps": target_fps,
        "real_fps": round(float(real_fps), 2),
        "mean_dt_ms": round(mean_dt_ms, 2),
        "p95_dt_ms": round(p95_dt_ms, 2),
        "p99_dt_ms": round(p99_dt_ms, 2),
        "brightness_drift": {idx: round(val, 2) for idx, val in drift_by_cam.items()},
        "fps_ok": real_fps >= target_fps * 0.90,
        "jitter_ok": p99_dt_ms <= 80.0,
    }


def run_camera_stress_test(
    cam_indices: list[int],
    width: int,
    height: int,
    fps: int,
    duration: int,
    scan_only: bool = False,
) -> int:
    """Run serial scan and OpenCV camera stress test; return 0 on pass, 1 on failure."""
    if len(set(cam_indices)) != len(cam_indices):
        print(f"[ERROR] Índices de cámara duplicados: {cam_indices}. Usa índices distintos (ej. --cameras 0 1).")
        return 1

    print("=" * 78)
    print("  PASO 4.2: DIAGNÓSTICO DE HARDWARE (PUERTOS SERIE Y CÁMARAS A 30 FPS)")
    print("=" * 78)

    scan_serial_ports()
    if scan_only:
        print("\n[•] Modo --scan-only activo: omitiendo captura continua de vídeo.")
        print("=" * 78)
        return 0

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) no está instalado.")
        return 1

    # -------------------------------------------------------------------------
    # [MARCA 2: APERTURA Y CONFIGURACIÓN UVC (COMPRESIÓN MJPG)]
    # -------------------------------------------------------------------------
    print(f"\n[MARCA 2] Abriendo cámara(s) OpenCV índice(s)={cam_indices} ({width}x{height} @ {fps} FPS)...")
    caps = {}
    for idx in cam_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"          [!] No se pudo abrir la cámara #{idx} (revisa permisos de Cámara en macOS o conexión).")
            for c in caps.values():
                c.release()
            return 1

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"          • Cámara #{idx} negociada a : {actual_w}x{actual_h} @ {actual_fps:.1f} FPS nominales")
        caps[idx] = cap

    # -------------------------------------------------------------------------
    # [MARCA 3: CALENTAMIENTO DEL SENSOR (WARMUP)]
    # -------------------------------------------------------------------------
    print("\n[MARCA 3] Descartando primeros 15 frames de calentamiento del sensor...")
    last_frames = {}
    for _ in range(15):
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frames[idx] = frame

    snap_dir = Path("outputs/viz")
    snap_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in last_frames.items():
        snap_path = snap_dir / f"camera_{idx}_snapshot.jpg"
        cv2.imwrite(str(snap_path), frame)
        print(f"          • Snapshot cámara #{idx} guardado en: {snap_path} (shape={frame.shape})")

    # -------------------------------------------------------------------------
    # [MARCA 4: MEDICIÓN DE JITTER TEMPORAL (P95 / P99) Y AUTO-EXPOSICIÓN]
    # -------------------------------------------------------------------------
    print(f"\n[MARCA 4] Midiendo estabilidad de captura continua durante {duration} segundos...")
    timestamps: list[float] = []
    brightness_means: dict[int, list[float]] = {idx: [] for idx in cam_indices}
    dropped_reads = 0

    t_start = time.perf_counter()
    while (time.perf_counter() - t_start) < duration:
        t_frame = time.perf_counter()
        all_ok = True
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if not ret or frame is None:
                all_ok = False
                dropped_reads += 1
            else:
                brightness_means[idx].append(float(np.mean(frame)))
        if all_ok:
            timestamps.append(t_frame)

    for cap in caps.values():
        cap.release()

    if len(timestamps) < 5:
        print("[ERROR] No se capturaron suficientes frames válidos.")
        return 1

    metrics = compute_stream_metrics(timestamps, brightness_means, dropped_reads, fps)
    target_ms = 1000.0 / fps

    print("\n--- Resultados del Stress Test de Cámara(s) ---")
    print(f"  • Frames sincronizados  : {metrics['frames']} (Lecturas fallidas: {metrics['dropped_reads']})")
    print(f"  • FPS reales sostenidos : {metrics['real_fps']:.2f} FPS (Objetivo: {fps} FPS)")
    print(
        f"  • Latencia entre frames : media={metrics['mean_dt_ms']:.1f}ms | "
        f"P95={metrics['p95_dt_ms']:.1f}ms | P99={metrics['p99_dt_ms']:.1f}ms "
        f"(Ideal: {target_ms:.1f}ms)"
    )

    for idx, b_drift in metrics["brightness_drift"].items():
        print(
            f"  • Estabilidad de Brillo (Cámara #{idx}): variación={b_drift:.2f}/255 "
            f"({'ESTABLE' if b_drift < 10 else 'ALERTA: Auto-Exposición fluctuando'})"
        )

    if metrics["fps_ok"] and metrics["jitter_ok"]:
        print("\n  ✅ [OK] Captura de vídeo lista para teleoperar y grabar en LeRobot.")
    else:
        print("\n  ⚠️ [ATENCIÓN] FPS bajos o jitter P99 > 80ms.")
        print("     Guía Bilbao: conecta al menos una cámara directa al portátil (no ambas al mismo hub)")
        print("     y usa un hub USB con alimentación propia.")
    print("=" * 78)
    return 0 if metrics["fps_ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnóstico de puertos serie y cámaras USB para LeRobot.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0], help="Índices de cámaras (ej. 0 o 0 1)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--scan-only", action="store_true", help="Solo escanear puertos serie sin abrir cámaras.")
    args = parser.parse_args()
    raise SystemExit(
        run_camera_stress_test(args.cameras, args.width, args.height, args.fps, args.duration, scan_only=args.scan_only)
    )
