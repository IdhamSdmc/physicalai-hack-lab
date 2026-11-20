# Cheat Sheet Oficial de Supervivencia: 48h en el Hackathon LeRobot Bilbao (LeRobot v0.7.0 / SO-101)

Este manual contiene la secuencia exacta de comandos para **LeRobot v0.7.0** (compatible con nuestro entorno local v0.6.2+ en Apple M5) alineada con la [Guía oficial del SO-101](https://hackathonbilbao.eus/guia-so-101) para el **viernes 20 de noviembre a las 16:00 en La Perrera Espazioa (Sabino Arana 50, Bilbao)**.

> [!CAUTION]
> **Cambios Críticos de LeRobot v0.7.0 y Reglas de Seguridad SO-101:**
> 1. **`lerobot-record` vs `lerobot-rollout`:** `lerobot-record` es solo para grabar demostraciones humanas; **`lerobot-rollout`** despliega el modelo en el robot físico.
> 2. **Prefijo `rollout_` obligatorio en `lerobot-rollout`:** Todo dataset grabado con `lerobot-rollout` (`episodic`, `dagger`, `highlight`, `sentry`) **debe empezar por `rollout_`** tras `<usuario>/` (por ejemplo `${ROLLOUT_DATASET_REPO}` o `${DAGGER_DATASET_REPO}`).
> 3. **Sufijo de fecha automático:** `lerobot-record` añade `_AAAAMMDD_HHMMSS` al nombre del dataset salvo que pases `--dataset.no_stamp=true`.
> 4. **Alimentación eléctrica:** Etiqueta las fuentes (**5 V Leader** vs **12 V Follower**), nunca las intercambies y **quita la corriente antes de tocar cables de servos**.

---

## FASE 0: Minuto 1 en la Mesa — Fijación Física, Permisos y Preflight (`Viernes 16:00`)

1. **Fija la base del Follower y los trípodes de las cámaras** (`top` y `wrist`). Si una cámara se mueve 2 cm a mitad del hackathon, los episodios previos pierden consistencia espacial.
2. **Mantén el brazo Leader fuera del campo visual (FOV) de las cámaras** durante las grabaciones.
3. **Topología USB:** Usa un **hub USB con alimentación propia** y **conecta al menos una cámara directa al MacBook** (no ambas cámaras en el mismo hub USB 2.0).
4. **Permiso macOS:** Da a Terminal permiso de **Monitorización de entrada** (*Ajustes del Sistema -> Privacidad y Seguridad*) para que funcionen las teclas `→`/`←`/`Esc` y `Space`/`Tab` en DAgger.
5. Descubre puertos/cámaras y ejecuta el preflight antes de calibrar:

```bash
source .venv/bin/activate

# Paso 0.1: Descubrir qué puerto es Follower y cuál es Leader
lerobot-find-port

# Paso 0.2: Descubrir índices de cámara (guarda capturas en outputs/captured_images)
lerobot-find-cameras opencv

# Paso 0.3: Stress Test de 10 segundos a tus 2 cámaras USB (confirmar 30 FPS sin caídas)
python scripts/04_camera_stress_test.py --cameras 0 1 --duration 10
```

6. Edita [`scripts/env_hackathon.sh`](scripts/env_hackathon.sh) con tus puertos e índices de cámara, cárgalo y valida todo con [`scripts/09_preflight_so101.py`](scripts/09_preflight_so101.py):

```bash
source scripts/env_hackathon.sh
python scripts/09_preflight_so101.py --check-hardware
```

> **Alternativas sin terminal (guía oficial):** Puedes usar **LeLab** (`uv tool install git+https://github.com/huggingface/leLab.git && lelab`) o **SenseCraft Robotics** (app de escritorio macOS/Windows). Si subes datasets/modelos o lanzas HF Jobs, autentícate antes con `hf auth login`.

---

## FASE 1: Calibración y Teleoperación de Prueba (`Viernes noche`)

> [!IMPORTANT]
> **Procedimiento oficial de calibración SO-101 ([guía Hugging Face](https://huggingface.co/docs/lerobot/so101)):**
> 1. **Postura media (Homing):** Coloca todas las articulaciones exactamente a **mitad de recorrido** (siguiendo el diagrama que imprime LeRobot) y pulsa `Enter` — fija el centro en `2047` ticks.
> 2. **Recorrido completo:** Mueve una a una las **5 articulaciones con tope** (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `gripper`) de extremo a extremo y pulsa `Enter` (`wrist_roll` queda automáticamente en vuelta completa `[0..4095]`).
> 3. **IDs únicos e invariables:** Usa siempre los mismos `--robot.id` / `--teleop.id` y **nunca** le des al Leader el ID del Follower. Si ya existía calibración y quieres rehacerla, pulsa `c` + `Enter`.

```bash
# 1.1 Calibrar brazo Esclavo (Follower)
lerobot-calibrate \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID}

# 1.2 Calibrar brazo Maestro (Leader)
lerobot-calibrate \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID}

# 1.3 Verificar calibraciones guardadas (6 motores, homing centrado y sin duplicado Leader=Follower)
python scripts/09_preflight_so101.py --require-calibration

# 1.4 Probar Teleoperación en vivo con cámaras (sin grabar aún)
lerobot-teleoperate \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --display_data=true
```

> **Si un servo no responde o falla la ID en un motor sustituido:** En placas Waveshare verifica que los **dos jumpers están en el canal `B` (USB)**. Si hace falta reconfigurar IDs/baudrate, usa `lerobot-setup-motors --robot.type=so101_follower --robot.port=${FOLLOWER_PORT}` (o `--teleop.type=so101_leader --teleop.port=${LEADER_PORT}`) conectando **un único motor cada vez** desde `gripper` (id 6) hasta `shoulder_pan` (id 1).

---

## FASE 2: El "Smoke Test" de 5 Episodios (`Viernes noche`)

**Regla de oro:** Valida el flujo completo de principio a fin el viernes por la noche con **5 episodios** en posición fija antes de grabar 50.

### 2.1 Grabar 5 episodios (con `--dataset.no_stamp=true` para nombre exacto)
- **Atajos de teclado durante `lerobot-record`:**
  - `→` o `n`: Guardar episodio actual y pasar al siguiente (reset).
  - `←` o `r`: Descartar y repetir el episodio actual si salió mal.
  - `Esc` o `q`: Terminar sesión (y subir al Hub si `--dataset.push_to_hub=true`).
- **Recuerda la Lección 4:** Empieza a mover el Leader en el instante en que arranca el episodio (evita tiempos muertos iniciales >1.5s) y empieza cada episodio desde la misma postura inicial.

```bash
lerobot-record \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=8 \
  --dataset.no_stamp=true \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

### 2.2 Auditar tus episodios en Parquet (< 2 segundos) y guardar reporte
```bash
python scripts/02_audit_dataset.py \
  --repo-id ${DATASET_REPO} \
  --local-dir ~/.cache/huggingface/lerobot/${DATASET_REPO} \
  --json-out outputs/audit_report.json
```

### 2.3 Reproducir un episodio grabado en el brazo físico (`lerobot-replay`)
```bash
lerobot-replay \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.episode=0
```

---

## FASE 3: Captura del Dataset Completo y Limpieza (`Sábado`)

Referencias oficiales de la guía SO-101 / Seeed:
- **Nivel Inicial** (coger en A y dejar en B, posiciones fijas): **~20 episodios**, `10.000–15.000` pasos ACT.
- **Nivel Intermedio** (posiciones aleatorias, ~10 ep. por posición): **~50 episodios**, `30.000–40.000` pasos ACT.
- **Nivel Avanzado** (varios pasos): **~100 episodios**, `80.000–100.000` pasos ACT.

### 3.1 Continuar grabando sobre el mismo dataset (`--resume=true`)
> **OJO v0.7.0:** Con `--resume=true`, `--dataset.num_episodes` indica **solo los episodios nuevos que añades** (no el total acumulado). Si no usaste `--dataset.no_stamp=true`, pasa el nombre completo con `_AAAAMMDD_HHMMSS` (`ls ~/.cache/huggingface/lerobot/${HF_USER}`).

```bash
lerobot-record \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=~/.cache/huggingface/lerobot/${DATASET_REPO} \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes=15 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true \
  --resume=true
```

### 3.2 Auditar, eliminar episodios defectuosos y publicar al Hub (+3 pts bonus)
Si `02_audit_dataset.py` detecta episodios malos (anota cuántos grabaste y cuántos descartaste para el jurado):
- **Opción A (al entrenar):** `--dataset.exclude_episodes="[12, 27]"`
- **Opción B (limpiar el dataset):**
  ```bash
  lerobot-edit-dataset \
    --repo_id ${DATASET_REPO} \
    --root ~/.cache/huggingface/lerobot/${DATASET_REPO} \
    --operation.type delete_episodes \
    --operation.episode_indices "[12, 27]"
  ```

---

## FASE 4: Entrenamiento ACT (Local M5 vs Colab / HF Jobs) (`Sábado`)

> [!NOTE]
> **Estrategia realista en Apple M5 (`~1.65 s/paso` con `batch_size=8`):**
> - **Prueba rápida en local M5:** `1.000–2.000` pasos (~28–55 min).
> - **Entrenamiento largo (`20.000–40.000` pasos):** En M5 tarda ~9–18 h (útil de noche); para iterar rápido de día, sube el dataset al Hub (`hf auth login`) y entrena en **Google Colab** o **Hugging Face Jobs** (`--job.target=a10g-small`).

```bash
# Entrenamiento local en MacBook M5 (guarda checkpoints cada 500 o 2000 pasos)
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.repo_id=${POLICY_REPO} \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --dataset.image_transforms.enable=true \
  --batch_size=8 \
  --num_workers=2 \
  --steps=10000 \
  --log_freq=50 \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --output_dir=outputs/train/act_bilbao_main \
  --trackio.enable=false
```

- **Si clasificas piezas por color:** desactiva `hue` o revisa primero las augmentations con `lerobot-imgtransform-viz` (`scripts/07_visualize_episode_and_augmentations.py`).
- **Retomar un entrenamiento cortado:**
  ```bash
  lerobot-train \
    --config_path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
  ```

> [!TIP]
> **Suavizar temblores en inferencia:** Añade `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` (tu M5 infiere en `18.5 ms`, soportando 30 Hz con Temporal Ensembling).

---

## FASE 5: Despliegue y Evaluación en el Robot (`lerobot-rollout`)

En **v0.7.0**, `--device` se detecta automáticamente o puede forzarse (`--device=mps`, `cuda`, `cpu`). Si pruebas un checkpoint intermedio de Hub, usa `--policy.pretrained_revision=020000`. Para VLAs lentos (`SmolVLA`, `π0`, `π0.5`), añade `--inference.type=rtc`.

### 5.1 Modo `base` — Demo en vivo y pruebas rápidas (sin grabar)
```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model \
  --device=${HACK_DEVICE} \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK_DESC}" \
  --display_data=true \
  --duration=60
```

### 5.2 Modo `episodic` — Ensayar los 5 intentos seguidos del Jurado
Mide la tasa de acierto (objetivo guía: **3/5 o más**) grabando en un dataset con prefijo `rollout_`:
```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model \
  --device=${HACK_DEVICE} \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --dataset.repo_id=${ROLLOUT_DATASET_REPO} \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --dataset.single_task="${TASK_DESC}"
```

### 5.3 Modo `dagger` — Corrección humana en vivo (prefijo `rollout_` obligatorio)
```bash
lerobot-rollout \
  --strategy.type=dagger \
  --policy.path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model \
  --device=${HACK_DEVICE} \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --dataset.repo_id=${DAGGER_DATASET_REPO} \
  --dataset.push_to_hub=false \
  --dataset.single_task="${TASK_DESC}" \
  --duration=120
```
*(Otros modos de v0.7.0: `highlight` guarda los últimos segundos al pulsar tecla; `sentry` graba continuo con subidas periódicas).*

---

## FASE 6: Evaluación del Jurado y Entregas (`Domingo`)

| Criterio | Qué valora el jurado | Puntos |
| :--- | :--- | :---: |
| **Demo en vivo** | **5 intentos seguidos** de la tarea completa sin tocar el robot (6 pts/acierto). | **30** |
| **Datos y entrenamiento** | Cantidad, variedad, consistencia, encuadre e iteración sobre fallos (lleva el resumen de `02_audit_dataset.py --json-out`). | **20** |
| **Dificultad de la tarea** | Ambición realista para 48h (tolerancia a cambios de posición o luz). | **15** |
| **Utilidad e impacto** | Si resuelve un problema real. | **15** |
| **Innovación e ingenio** | Originalidad del enfoque/solución (ej. orquestador multi-política). | **10** |
| **Presentación y equipo** | Pitch de **3 minutos** + respuestas al jurado + **vídeo de 1 minuto**. | **10** |
| **Bonus: Dataset + Repo** | Dataset publicado en Hugging Face (**+3**) + repositorio reproducible (**+2**). | **+5** |

**Checklist antes del pitch:**
1. Grabar **vídeo de 1 minuto** mostrando la tarea funcionando.
2. Publicar el dataset limpio en Hugging Face (`+3 pts`) y verificar `python -m unittest discover -s tests -v` + README reproducible (`+2 pts`).
3. Dejar la mesa despejada, el Leader fuera del FOV y lanzar `python scripts/09_preflight_so101.py --check-hardware` antes de los 5 intentos en vivo.
