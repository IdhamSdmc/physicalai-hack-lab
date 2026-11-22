# Panel de Control Web Dual-View — SO-101 Zero-Cross-Contamination Triage

Hackathon LeRobot Bilbao II · 20–22 nov 2026 · La Perrera Espazioa

Interfaz web móvil/iPad y desktop (`React + Vite + Three.js` sobre `FastAPI`) para operar de forma remota la celda robótica SO-101 con detección OpenCV en tiempo real y gemelo 3D sincronizado.

## Arquitectura y Enlaces
- **Repositorio de Robótica / Supervisor / Datasets:** [`IdhamSdmc/physicalai-hack-lab`](https://github.com/IdhamSdmc/physicalai-hack-lab)
- **Datasets Hugging Face Hub:** [`coalva94/bilbao_triage_complete_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_complete_v3) · [`coalva94/bilbao_triage_sano_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_sano_v3) · [`coalva94/bilbao_triage_contaminado_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_contaminado_v3)
- **Modelos ACT V3:** [`coalva94/act_sano_v3`](https://huggingface.co/coalva94/act_sano_v3) · [`coalva94/act_contaminado_v3`](https://huggingface.co/coalva94/act_contaminado_v3)

## Características del Panel
- **Vista Dual en Vivo:** cámara cenital OpenCV (`ZONA_A`, `ZONA_B_SUP` Almacén, `ZONA_B_INF` Desechos, `VASO_D`) junto a la escena 3D `Three.js` sincronizada.
- **Control Remoto Seguro (`LAN` / iPad):** botones **Iniciar**, **Detener** (envía `SIGTERM` para liberar torque de forma limpia) y métricas/logs en tiempo real vía `/api/status` y `/api/hud.jpg`.
- **Modo Simulación o Real (`config.yaml`):** copia [`config.example.yaml`](config.example.yaml) a `config.yaml` para elegir `mock_mode: true` o el supervisor real [`10_opencv_triage_supervisor.py`](https://github.com/IdhamSdmc/physicalai-hack-lab/blob/feat/so101-guide-validation/scripts/10_opencv_triage_supervisor.py).

## Puesta en Marcha Local
```bash
# Frontend (Vite + React)
npm install
npm run build

# Backend FastAPI (sirve dist/ y API de control)
cp config.example.yaml config.yaml
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Abrir `http://localhost:8080` en local o `http://<IP_MAC>:8080` desde el iPad en la misma red Wi-Fi.
