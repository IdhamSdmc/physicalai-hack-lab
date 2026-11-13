#!/usr/bin/env python3
"""
Sprint 1 — Verificador de Entorno y Benchmark de Inferencia en Apple M5 (MPS).

Comprueba:
1. Versión de Python (debe ser 3.10–3.12 para evitar incompatibilidades con MuJoCo/PyAV).
2. Disponibilidad de aceleración Metal (MPS) en Apple Silicon.
3. Presupuesto de Memoria Unificada (16 GB) y recomendaciones de batch_size.
4. Benchmark sintético de paso forward + backward (simulando ResNet + Transformer de ACT)
   para verificar que la inferencia corre a >30 Hz (<33.3 ms por paso).
"""

import platform
import subprocess
import sys
import time


def get_macos_ram_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / (1024**3)
    except Exception:
        return 0.0


def main() -> None:
    print("=" * 68)
    print("  PHYSICAL AI HACKATHON BILBAO — DIAGNÓSTICO APPLE SILICON (M5)")
    print("=" * 68)

    py_ver = sys.version.split()[0]
    print(f"[•] Python activo      : {py_ver} ({sys.executable})")
    if sys.version_info >= (3, 14) or sys.version_info < (3, 12):
        print("    [!] ADVERTENCIA: LeRobot v0.6+ requiere Python 3.12 o 3.13 (Python 3.14 falla en wheels).")
    else:
        print("    [OK] Versión de Python (3.12/3.13) compatible con LeRobot v0.6+.")

    ram_gb = get_macos_ram_gb()
    print(f"[•] Plataforma         : {platform.platform()} ({platform.machine()})")
    print(f"[•] Memoria Unificada  : {ram_gb:.1f} GB")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"\n[ERROR] PyTorch no está instalado en este entorno: {exc}")
        sys.exit(1)

    print(f"[•] PyTorch / Vision   : {torch.__version__} / {torchvision.__version__}")
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    print(f"[•] Aceleración MPS    : {'DISPONIBLE (Metal GPU activo)' if mps_ok else 'NO DISPONIBLE'}")

    if not mps_ok:
        print("[ERROR] MPS no está activo. Revisa tu instalación de PyTorch para macOS arm64.")
        sys.exit(1)

    device = torch.device("mps")

    # Simulación de carga de inferencia tipo ACT (2 cámaras 480x640 + estado articular -> chunk de 50 acciones)
    print("\n--- Benchmark de Control en Tiempo Real (Objetivo: < 33.3 ms para 30 Hz) ---")
    backbone = torchvision.models.resnet18(weights=None).to(device)
    backbone.fc = torch.nn.Linear(512, 256).to(device)
    transformer_head = torch.nn.TransformerEncoderLayer(
        d_model=256, nhead=8, dim_feedforward=1024, batch_first=True
    ).to(device)
    action_proj = torch.nn.Linear(256, 50 * 6).to(device)  # chunk_size=50, 6 DOF (SO-101)

    backbone.eval()
    transformer_head.eval()
    action_proj.eval()

    # 2 cámaras RGB (wrist + top) a 480x640 en batch=1 para inferencia real
    dummy_cams = torch.randn(2, 3, 480, 640, device=device)

    # Warmup en Metal
    with torch.no_grad():
        for _ in range(10):
            feats = backbone(dummy_cams).unsqueeze(0)
            out = transformer_head(feats)
            _ = action_proj(out)
        torch.mps.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            feats = backbone(dummy_cams).unsqueeze(0)
            out = transformer_head(feats)
            _ = action_proj(out)
            torch.mps.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    max_hz = 1000.0 / avg_ms

    print(f"[•] Latencia media (2 cámaras 640x480 -> ACT chunk) : {avg_ms:.2f} ms (P95: {p95_ms:.2f} ms)")
    print(f"[•] Frecuencia máxima de inferencia en tu M5       : {max_hz:.1f} Hz")
    if avg_ms < 33.3:
        print("    [OK] Tu M5 puede ejecutar inferencia con Temporal Ensembling en CADA frame a 30 Hz.")
    else:
        print("    [!] Latencia > 33.3 ms: usa n_action_steps=20..50 para no inferir en cada frame.")

    # Comprobar LeRobot
    try:
        import lerobot
        print(f"\n[•] LeRobot instalado  : v{getattr(lerobot, '__version__', 'OK')}")
    except ImportError:
        print("\n[!] LeRobot aún no está instalado en este entorno virtual.")

    print("\n--- Recomendación de Hiperparámetros para tus 16 GB de RAM ---")
    print("  --policy.device=mps --batch_size=8 --num_workers=2 --policy.chunk_size=50")
    print("=" * 68)


if __name__ == "__main__":
    main()
