#!/usr/bin/env python3
"""
================================================================================
PASO 4.2 — DIAGNÓSTICO DE PUERTOS SERIE Y STRESS TEST DE CÁMARAS USB (macOS)
================================================================================
[¿PARA QUÉ SIRVE ESTE SCRIPT EN EL DÍA CERO DEL HACKATHON?]
El 90% de los problemas de las primeras 3 horas en un hackathon de LeRobot son físicos:
  1. ¿Qué cable USB corresponde al brazo Leader (maestro) y cuál al Follower (esclavo)?
  2. ¿Qué índice de cámara (0, 1, 2) es la cámara de la pinza (`wrist/side`), cuál es
     la cámara cenital (`up/top`) y cuál es la webcam integrada del MacBook?
  3. ¿Pueden las cámaras mantener 30 FPS clavados (33.3 ms por frame) sin saturar
     el bus USB ni cambiar el brillo solas (Auto-Exposure)?
================================================================================
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np


def scan_serial_ports() -> None:
    print("\n[MARCA 1] Escaneando Puertos Serie USB (donde se conectan los brazos SO-101)...")
    print("          • En Linux se llaman : /dev/ttyACM0 o /dev/ttyUSB0")
    print("          • En macOS se llaman : /dev/tty.usbmodem* (ej. /dev/tty.usbmodem585A0076841)")
    try:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        if not ports:
            print("          • Puertos detectados : Ninguno conectado ahora mismo.")
        else:
            for p in ports:
                is_robot = "usbmodem" in p.device or "usbserial" in p.device
                tag = " <-- [PLACA SERVOS SO-101]" if is_robot else " (Puerto interno macOS)"
                print(f"          • {p.device:<32} | desc='{p.description}'{tag}")
    except ImportError:
        print("          [!] pyserial no disponible.")

    print("          💡 [TRUCO EN BILBAO]: Ejecuta '.venv/bin/lerobot-find-port'. Te pedirá desconectar")
    print("             el cable USB de un brazo y te dirá por diferencia qué puerto exacto era.")


def run_camera_stress_test(cam_indices: list[int], width: int, height: int, fps: int, duration: int) -> None:
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) no está instalado.")
        sys.exit(1)

    print("=" * 78)
    print("  PASO 4.2: DIAGNÓSTICO DE HARDWARE (PUERTOS SERIE Y CÁMARAS A 30 FPS)")
    print("=" * 78)

    scan_serial_ports()

    # -------------------------------------------------------------------------
    # [MARCA 2: APERTURA Y CONFIGURACIÓN UVC (COMPRESIÓN MJPG)]
    # ¿Por qué forzamos MJPG y 640x480?:
    # Un puerto USB 2.0 (el que llevan casi todas las cámaras web de robótica) tiene
    # un límite real de ~35 MB/s. Dos cámaras enviando 1080p sin comprimir (YUYV)
    # saturan el bus USB y una de las dos se congela o cae a 10 FPS. Con MJPG a
    # 640x480, puedes conectar hasta 3 cámaras a 30 FPS sin perder un solo frame.
    # -------------------------------------------------------------------------
    print(f"\n[MARCA 2] Abriendo cámara(s) OpenCV índice(s)={cam_indices} ({width}x{height} @ {fps} FPS)...")
    caps = {}
    for idx in cam_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"          [!] No se pudo abrir la cámara #{idx} (revisa permisos de Cámara en macOS o conexión).")
            for c in caps.values():
                c.release()
            return

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
    # Las cámaras web tardan ~0.5 segundos al encenderse en estabilizar el sensor CMOS.
    # -------------------------------------------------------------------------
    print("\n[MARCA 3] Descartando primeros 15 frames de calentamiento del sensor...")
    last_frames = {}
    for _ in range(15):
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frames[idx] = frame

    # Guardar una captura de referencia para saber qué cámara es cuál
    snap_dir = Path("outputs/viz")
    snap_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in last_frames.items():
        snap_path = snap_dir / f"camera_{idx}_snapshot.jpg"
        cv2.imwrite(str(snap_path), frame)
        print(f"          • Snapshot cámara #{idx} guardado en: {snap_path} (shape={frame.shape})")

    # -------------------------------------------------------------------------
    # [MARCA 4: MEDICIÓN DE JITTER TEMPORAL (P95 / P99) Y AUTO-EXPOSICIÓN]
    # LeRobot exige que los timestamps avancen cada ~33.3 ms. Si el percentil 99 (P99)
    # sube a >80 ms, `lerobot-record` lanzará warnings o corromperá el episodio.
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
        return

    dts_ms = np.diff(timestamps) * 1000.0
    real_fps = len(timestamps) / (timestamps[-1] - timestamps[0])
    target_ms = 1000.0 / fps

    print("\n--- Resultados del Stress Test de Cámara(s) ---")
    print(f"  • Frames sincronizados  : {len(timestamps)} (Lecturas fallidas: {dropped_reads})")
    print(f"  • FPS reales sostenidos : {real_fps:.2f} FPS (Objetivo: {fps} FPS)")
    print(
        f"  • Latencia entre frames : media={np.mean(dts_ms):.1f}ms | "
        f"P95={np.percentile(dts_ms, 95):.1f}ms | P99={np.percentile(dts_ms, 99):.1f}ms "
        f"(Ideal: {target_ms:.1f}ms)"
    )

    for idx, b_vals in brightness_means.items():
        b_drift = float(np.max(b_vals) - np.min(b_vals))
        print(
            f"  • Estabilidad de Brillo (Cámara #{idx}): variación={b_drift:.2f}/255 "
            f"({'ESTABLE' if b_drift < 10 else 'ALERTA: Auto-Exposición fluctuando'})"
        )

    if real_fps >= fps * 0.90:
        print("\n  ✅ [OK] Captura de vídeo lista para teleoperar y grabar en LeRobot.")
    else:
        print("\n  ⚠️ [ATENCIÓN] Los FPS reales están por debajo del objetivo. Revisa resolución o hub USB.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnóstico de puertos serie y cámaras USB para LeRobot.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0], help="Índices de cámaras (ej. 0 o 0 1)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=5)
    args = parser.parse_args()
    run_camera_stress_test(args.cameras, args.width, args.height, args.fps, args.duration)
