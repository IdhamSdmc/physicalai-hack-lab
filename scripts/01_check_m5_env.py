#!/usr/bin/env python3
"""
Sprint 1 — Verificador de Entorno y Benchmark de Inferencia en Apple Silicon (M5).

Comprueba (alineado con la guía SO-101 del Hackathon LeRobot Bilbao):
1. Versión de Python (3.12 o 3.13 para LeRobot v0.7.0).
2. Versión de macOS (>= 15 requerido para entrenar con MPS) y Memoria Unificada (16 GB).
3. Espacio libre en disco (>= 20 GB recomendados para vídeos de episodios).
4. Disponibilidad de aceleración Metal (MPS) en Apple Silicon.
5. Versión de LeRobot (v0.7.0 recomendada), binarios CLI, PyAV/ffmpeg y token HF Hub.
6. Benchmark sintético de inferencia tipo ACT (2 cámaras 640x480 -> chunk 50x6)
   para verificar >30 Hz (<33.3 ms por paso).
"""

import argparse
from importlib import metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

REQUIRED_LEROBOT_CLIS = (
    "lerobot-find-port",
    "lerobot-find-cameras",
    "lerobot-calibrate",
    "lerobot-teleoperate",
    "lerobot-record",
    "lerobot-replay",
    "lerobot-train",
    "lerobot-rollout",
    "lerobot-edit-dataset",
)


def get_macos_ram_gb() -> float:
    """Return physical RAM in GiB on macOS, or 0.0 when unavailable."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / (1024**3)
    except Exception:
        return 0.0


def parse_semver_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a leading numeric semantic version string into an integer tuple."""
    parts: list[int] = []
    for token in version_str.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_python_compatibility(version_info: tuple[int, ...] | None = None) -> tuple[bool, str]:
    """Check whether Python version matches LeRobot v0.6+/v0.7.0 requirements (3.12–3.13)."""
    vi = version_info or sys.version_info[:3]
    if vi >= (3, 14) or vi < (3, 12):
        return False, "ADVERTENCIA: LeRobot v0.7.0 requiere Python 3.12 o 3.13 (guía Bilbao usa 3.12)."
    return True, "Versión de Python (3.12/3.13) compatible con LeRobot v0.7.0."


def check_macos_mps_support(system_name: str | None = None, mac_ver: str | None = None) -> tuple[bool, str]:
    """Verify macOS >= 15 when running on Darwin (required for PyTorch MPS training)."""
    sys_name = system_name or platform.system()
    if sys_name != "Darwin":
        return True, f"Sistema operativo: {sys_name}"
    ver_str = mac_ver if mac_ver is not None else platform.mac_ver()[0]
    parsed = parse_semver_tuple(ver_str)
    if not parsed:
        return True, "macOS detectado (versión exacta no reportada)."
    if parsed[0] < 15:
        return False, f"macOS {ver_str} (<15): en macOS 14 el entrenamiento cae a CPU."
    return True, f"macOS {ver_str} (>=15 compatible con MPS)."


def check_disk_space_gb(target_path: Path | str = ".", min_gb: float = 20.0) -> tuple[bool, float]:
    """Return (meets_min_gb, free_gb) for the filesystem containing target_path."""
    usage = shutil.disk_usage(Path(target_path))
    free_gb = usage.free / (1024**3)
    return free_gb >= min_gb, free_gb


def check_video_toolchain() -> dict[str, Any]:
    """Check presence of `ffmpeg` in PATH and `av` (PyAV) Python package."""
    ffmpeg_path = shutil.which("ffmpeg")
    try:
        import av

        pyav_ver = getattr(av, "__version__", "OK")
    except ImportError:
        pyav_ver = None
    return {"ffmpeg_path": ffmpeg_path, "pyav_version": pyav_ver}


def check_lerobot_installation(bin_dir: Path | None = None) -> dict[str, Any]:
    """Inspect installed LeRobot version, v0.7.0 alignment, and CLI entrypoints."""
    bdir = bin_dir or Path(sys.executable).parent
    try:
        ver = metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        try:
            import lerobot

            ver = getattr(lerobot, "__version__", "unknown")
        except ImportError:
            ver = None

    parsed = parse_semver_tuple(ver) if ver else ()
    is_v070_plus = parsed >= (0, 7, 0)
    missing_clis = [cli for cli in REQUIRED_LEROBOT_CLIS if not ((bdir / cli).exists() or shutil.which(cli))]
    return {
        "version": ver,
        "is_v070_plus": is_v070_plus,
        "missing_clis": missing_clis,
        "lelab_installed": shutil.which("lelab") is not None or (bdir / "lelab").exists(),
    }


def check_hf_auth_offline() -> dict[str, Any]:
    """Check whether a local Hugging Face token is configured without network calls."""
    try:
        from huggingface_hub import get_token

        token = get_token()
        return {"has_token": bool(token)}
    except Exception:
        return {"has_token": False}


def run_act_benchmark(device: Any, steps: int = 50, warmup: int = 10) -> tuple[float, float, float]:
    """Benchmark a synthetic ACT forward pass (2x 640x480 images -> 50x6 action chunk)."""
    import torch
    import torchvision

    backbone = torchvision.models.resnet18(weights=None).to(device)
    backbone.fc = torch.nn.Linear(512, 256).to(device)
    transformer_head = torch.nn.TransformerEncoderLayer(
        d_model=256, nhead=8, dim_feedforward=1024, batch_first=True
    ).to(device)
    action_proj = torch.nn.Linear(256, 50 * 6).to(device)  # chunk_size=50, 6 DOF (SO-101)

    backbone.eval()
    transformer_head.eval()
    action_proj.eval()

    dummy_cams = torch.randn(2, 3, 480, 640, device=device)

    def _sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            feats = backbone(dummy_cams).unsqueeze(0)
            out = transformer_head(feats)
            _ = action_proj(out)
        _sync()

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            feats = backbone(dummy_cams).unsqueeze(0)
            out = transformer_head(feats)
            _ = action_proj(out)
            _sync()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = sorted(latencies_ms)[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95))]
    max_hz = 1000.0 / avg_ms
    return avg_ms, p95_ms, max_hz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de entorno y latencia para el Hackathon SO-101 Bilbao.")
    parser.add_argument("--allow-cpu", action="store_true", help="No abortar si MPS/CUDA no está disponible (útil en CI/sandbox).")
    parser.add_argument("--skip-benchmark", action="store_true", help="Omitir benchmark sintético de inferencia ACT.")
    parser.add_argument("--min-disk-gb", type=float, default=20.0, help="Mínimo de GB libres recomendados para vídeos.")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("  PHYSICAL AI HACKATHON BILBAO — DIAGNÓSTICO PORTÁTIL & APPLE SILICON")
    print("=" * 72)

    py_ver = sys.version.split()[0]
    py_ok, py_msg = check_python_compatibility()
    print(f"[•] Python activo      : {py_ver} ({sys.executable})")
    print(f"    [{'OK' if py_ok else '!'}] {py_msg}")

    ram_gb = get_macos_ram_gb()
    mac_ok, mac_msg = check_macos_mps_support()
    disk_ok, free_gb = check_disk_space_gb(".", min_gb=args.min_disk_gb)
    print(f"[•] Plataforma         : {platform.platform()} ({platform.machine()})")
    print(f"    [{'OK' if mac_ok else '!'}] {mac_msg}")
    if ram_gb > 0:
        print(f"[•] Memoria Unificada  : {ram_gb:.1f} GB")
    print(f"[•] Disco libre        : {free_gb:.1f} GB ({'OK' if disk_ok else f'AVISO: < {args.min_disk_gb:.0f} GB recomendados'})")

    try:
        import torch
    except ImportError as exc:
        print(f"\n[ERROR] PyTorch no está instalado en este entorno: {exc}")
        return 1

    print(f"[•] PyTorch            : {torch.__version__}")
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cuda_ok = torch.cuda.is_available()
    print(f"[•] Aceleración MPS    : {'DISPONIBLE (Metal GPU activo)' if mps_ok else 'NO DISPONIBLE'}")
    if cuda_ok:
        print(f"[•] Aceleración CUDA   : DISPONIBLE ({torch.cuda.get_device_name(0)})")

    if not mps_ok and not cuda_ok and not args.allow_cpu:
        print("[ERROR] MPS/CUDA no está activo. Usa --allow-cpu en CI/sandbox o revisa PyTorch arm64.")
        return 1

    device = torch.device("mps" if mps_ok else ("cuda" if cuda_ok else "cpu"))

    if not args.skip_benchmark:
        print(f"\n--- Benchmark de Control en Tiempo Real en '{device.type}' (Objetivo: < 33.3 ms para 30 Hz) ---")
        avg_ms, p95_ms, max_hz = run_act_benchmark(device)
        print(f"[•] Latencia media (2 cámaras 640x480 -> ACT chunk) : {avg_ms:.2f} ms (P95: {p95_ms:.2f} ms)")
        print(f"[•] Frecuencia máxima de inferencia                 : {max_hz:.1f} Hz")
        if avg_ms < 33.3:
            print("    [OK] Puede ejecutar inferencia con Temporal Ensembling en CADA frame a 30 Hz.")
        else:
            print("    [!] Latencia > 33.3 ms: usa n_action_steps=20..50 para no inferir en cada frame.")

    video_info = check_video_toolchain()
    print("\n--- Herramientas LeRobot, Vídeo y Hugging Face (Guía Bilbao v0.7.0) ---")
    print(f"[•] PyAV (`av`)        : {video_info['pyav_version'] or 'NO INSTALADO'}")
    print(f"[•] Binario `ffmpeg`   : {video_info['ffmpeg_path'] or 'NO ENCONTRADO en PATH (PyAV sigue funcionando; guía recomienda ffmpeg)'}")

    lr_info = check_lerobot_installation()
    if lr_info["version"] is None:
        print("[!] LeRobot aún no está instalado en este entorno virtual.")
    else:
        tag = "OK (alineado con v0.7.0)" if lr_info["is_v070_plus"] else "AVISO: guía oficial usa v0.7.0 (pip install 'lerobot[core_scripts,training,feetech]==0.7.0')"
        print(f"[•] LeRobot instalado  : v{lr_info['version']} -> {tag}")
        if lr_info["missing_clis"]:
            print(f"    [!] CLIs ausentes  : {', '.join(lr_info['missing_clis'])}")
        else:
            print(f"    [OK] Todos los CLIs clave ({len(REQUIRED_LEROBOT_CLIS)}) disponibles.")
        print(f"[•] LeLab GUI          : {'Instalado' if lr_info['lelab_installed'] else 'No instalado (opcional: uv tool install git+https://github.com/huggingface/leLab.git)'}")

    hf_info = check_hf_auth_offline()
    print(f"[•] Token Hugging Face : {'Configurado' if hf_info['has_token'] else 'No detectado (ejecuta `hf auth login` para Colab/HF Jobs y bonus +3)'}")

    print("\n--- Recomendación de Entrenamiento (16 GB RAM Apple Silicon vs Nube) ---")
    print("  • Smoke local M5 : --policy.device=mps --batch_size=8 --num_workers=2 --steps=1000..2000")
    print("  • Entreno largo  : M5 (~1.65s/step -> 10k steps ~4.5h) o Colab / HF Jobs (--job.target=a10g-small)")
    print("  • macOS permiso  : Concede 'Monitorización de entrada' a Terminal para teclas de grabar/DAgger")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
