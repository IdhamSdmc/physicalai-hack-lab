# Cheat Sheet Oficial de Supervivencia: 48h en el Physical AI Hackathon Bilbao (LeRobot v0.6.2)

Este manual contiene la secuencia exacta de comandos para **LeRobot v0.6.2 en tu MacBook Apple M5**, en el orden cronológico en que los ejecutarás desde que te sientes en la mesa en La Perrera (Bilbao).

> [!CAUTION]
> **Cambio Crítico en LeRobot v0.6.2 que confunde a los equipos:**
> En tutoriales antiguos de internet (v0.4), para evaluar una política en el robot físico se usaba `lerobot-record --policy.path=...`.
> **En LeRobot v0.6.2 eso ya NO existe:**
> - `lerobot-record` es **exclusivamente** para grabar demostraciones humanas.
> - **`lerobot-rollout`** es el comando oficial para **desplegar y ejecutar tu modelo entrenado en el brazo físico** (e incluso permite hacer correcciones humanas en vivo con `--strategy.type=dagger`).

---

## FASE 0: Minuto 1 en la Mesa — Fijación Física y Variables (`Hora 0:00`)

1. **Pega con cinta americana o sargentos** la base del brazo *Follower* y el soporte de la cámara cenital (`up`) a la mesa. Si la cámara externa se mueve 2 cm a mitad del hackathon, todos los episodios anteriores quedan inutilizados.
2. Conecta las 2 cámaras USB en puertos USB-C distintos de tu MacBook M5 (evita meter ambas cámaras + ambos brazos en un único hub USB 2.0 barato).
3. Abre tu terminal en la raíz del repositorio (`physicalai-hack-lab`) y descubre tus puertos y cámaras:

```bash
source .venv/bin/activate

# Paso 0.1: Descubrir qué puerto /dev/tty.usbmodem* es el Follower y cuál el Leader
lerobot-find-port

# Paso 0.2: Descubrir qué índice (0, 1, 2) es cada cámara USB (guarda fotos en outputs/captured_images)
lerobot-find-cameras opencv

# Paso 0.3: Hacer Stress Test de 10 segundos a tus 2 cámaras USB para confirmar 30 FPS clavados
python scripts/04_camera_stress_test.py --cameras 0 1 --duration 10
```

4. Edita [`scripts/env_hackathon.sh`](scripts/env_hackathon.sh) con tus 2 puertos `/dev/tty.usbmodem*` y tus 2 índices de cámara, y cárgalo en tu terminal:
```bash
source scripts/env_hackathon.sh
```

---

## FASE 1: Calibración y Teleoperación de Prueba (`Hora 0:30 – 1:30`)

> **¿Para qué se calibra?** Para que cuando el brazo *Leader* esté a `0°`, el *Follower* también esté exactamente a `0°`. La calibración se guarda en `~/.cache/huggingface/lerobot/calibration/`.

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

# 1.3 Probar Teleoperación en vivo con cámaras (sin grabar aún)
# Practica mover el brazo suave, continuo y sin paradas durante 5 minutos.
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

---

## FASE 2: El "Smoke Test" de 5 Episodios (`Hora 1:30 – 3:00`)

**Regla de oro:** Nunca grabes 50 episodios sin antes probar el bucle completo con **5 episodios** dejando el objeto siempre en el mismo sitio.

### 2.1 Grabar 5 episodios locales (sin subir a internet: `--dataset.push_to_hub=false`)
- **Teclas durante `lerobot-record`:**
  - `Flecha Derecha (->)`: Terminar episodio actual con éxito y pasar al tiempo de reset.
  - `Flecha Izquierda (<-)`: Descartar y repetir el episodio actual si te equivocaste.
  - `Escape (ESC)`: Terminar la sesión de grabación.
- **Recuerda la Lección 4:** ¡Empieza a mover el brazo inmediatamente al arrancar el episodio! No dejes 2.5 segundos de tiempo muerto al inicio.

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
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

### 2.2 Auditar tus 5 episodios en Parquet (< 2 segundos)
```bash
python scripts/02_audit_dataset.py \
  --repo-id ${DATASET_REPO} \
  --local-dir ~/.cache/huggingface/lerobot/${DATASET_REPO}
```

### 2.3 Reproducir un episodio grabado en el brazo físico (`lerobot-replay`)
Para comprobar al 100% que los motores del *Follower* reproducen la grabación:
```bash
lerobot-replay \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.episode=0
```

---

## FASE 3: Captura del Dataset Completo (40–50 Episodios) y Limpieza (`Hora 3:00 – 12:00`)

### 3.1 Continuar grabando sobre el mismo dataset (`--resume=true`)
Divide la mesa en 5 posiciones distintas para el objeto y graba 8 repeticiones por posición (40 episodios) + 10 episodios de recuperación:
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
  --dataset.num_episodes=45 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true \
  --resume=true
```

### 3.2 Auditar y eliminar episodios defectuosos sin regrabar
Si `02_audit_dataset.py` te avisa de que los episodios `#12` y `#27` tuvieron *Hesitation* o *Frame Drops*, puedes:
- **Opción A (Instantánea al entrenar):** Pasar `--dataset.exclude_episodes="[12, 27]"` a `lerobot-train`.
- **Opción B (Borrarlos del dataset):**
  ```bash
  lerobot-edit-dataset \
    --repo_id ${DATASET_REPO} \
    --operation.type delete_episodes \
    --operation.episode_indices "[12, 27]"
  ```

---

## FASE 4: Entrenamiento 100% Local en tu Apple M5 (`Hora 12:00 – 24:00`)

Entrena **ACT** en tu GPU Metal (`mps`). Con `batch_size=8` y `--save_freq=2000`, tendrás checkpoints listos para probar cada ~40 minutos:

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=mps \
  --policy.push_to_hub=false \
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

> [!TIP]
> **¿Cómo suavizar el movimiento si el brazo tiembla durante la inferencia?**
> Activa **Temporal Ensembling** añadiendo `--policy.temporal_ensemble_coeff=0.01` y `--policy.n_action_steps=1` (tu M5 tarda solo `18.5 ms` por paso, así que puede inferir en cada frame a `30 Hz` promediando exponencialmente las predicciones).

---

## FASE 5: Despliegue Autónomo en el Robot Físico (`lerobot-rollout`)

En **LeRobot v0.6.2**, usa `lerobot-rollout` apuntando a tu carpeta `pretrained_model`:

### 5.1 Inferencia Autónoma Pura (`--strategy.type=base`)
```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model \
  --policy.device=mps \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK_DESC}" \
  --duration=60
```

### 5.2 Arma Secreta: Corrección Humana en Vivo DAgger (`--strategy.type=dagger`)
Si el robot autónomo falla en ciertas esquinas de la mesa, lanza `lerobot-rollout` en modo **`dagger`**: el robot se mueve solo con tu modelo ACT, y cuando veas que va a fallar, **tomas el brazo Leader y lo corriges en vivo**. Esas correcciones se guardan como un dataset nuevo para hacer *fine-tuning* (`--resume=true` o `--policy.path=...` en `lerobot-train`):

```bash
lerobot-rollout \
  --strategy.type=dagger \
  --policy.path=outputs/train/act_bilbao_main/checkpoints/last/pretrained_model \
  --policy.device=mps \
  --robot.type=${ROBOT_TYPE} \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=${TELEOP_TYPE} \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=${TELEOP_ID} \
  --dataset.repo_id=${DATASET_REPO}_dagger \
  --dataset.single_task="${TASK_DESC}" \
  --duration=120
```
