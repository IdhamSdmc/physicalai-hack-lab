# Prompt Maestro para el Agente en el Mac de Rodaje — Grabación Guiada de 64 Episodios (LeLab) + Entrenamiento Dual ACT + Calibrador y Supervisor OpenCV

> **Instrucciones de uso:** Copia todo el bloque de abajo (o dile a tu agente en el otro Mac: *"Lee y ejecuta el archivo `PROMPT_AGENTE_MAC_RODAJE_Y_OPENCV.md` paso a paso"*).

---

```markdown
Actúa como nuestro Ingeniero Principal de Robótica (LeRobot v0.7.0 + SO-101 + OpenCV) en este Mac donde tenemos montados los brazos SO-101 (Leader y Follower), las 2 cámaras (`top` cenital y `wrist` en muñeca) y LeLab.

## 1. CONTEXTO DEL RETO Y ARQUITECTURA GANADORA
Estamos construyendo **"Zero-Cross-Contamination Triage"** (Celda Robótica de Clasificación y Descontaminación con Bioseguridad) para ganar el Hackathon de LeRobot Bilbao:
- **Baremo del jurado (100 + 5 pts):**
  - **30 pts — Demo en vivo:** 5 intentos seguidos sin tocar el robot (6 pts por acierto). La fiabilidad 5/5 es la prioridad #1.
  - **20 pts — Datos y entrenamiento:** El jurado preguntará cuántos episodios grabamos, cuántos descartamos y qué cambiamos tras el primer entrenamiento.
  - **15 pts Dificultad | 15 pts Utilidad | 10 pts Innovación | 10 pts Presentación | +5 pts Bonus (Dataset en HF + Repo reproducible).**

### Escenario físico ("Media Luna")
- **Zona A (Izquierda — Zona de Inspección/Entrada):** Empieza con **4 cubos** en total: **2 cubos Verdes (`S` = Sanos)** y **2 cubos Rojos (`C` = Contaminados)**.
- **Zona B (Derecha — Zona de Clasificación):** Dividida por una **línea horizontal**:
  - **Zona B Superior (`B_SUP`):** Destino de los cubos **Verdes (`S` — Sanos)**.
  - **Zona B Inferior (`B_INF`):** Destino de los cubos **Rojos (`C` — Contaminados / Desechos)**.
- **Vaso D (Descontaminante):** Recipiente con líquido descontaminante simbólico.
- **Posición `HOME`:** Postura fija de inicio y fin de cada episodio donde el brazo NO tapa la vista de la cámara `top` sobre la `Zona A`.

### Regla de negocio y movimiento
1. **Protocolo Sano (`S` — Cubo Verde):**
   - El brazo sale de `HOME` -> recoge **1 cubo Verde** de `Zona A` -> lo deposita en **`Zona B Superior`** -> **NO pasa por el Vaso D** -> vuelve a `HOME`.
2. **Protocolo Contaminado (`C` — Cubo Rojo):**
   - El brazo sale de `HOME` -> recoge **1 cubo Rojo** de `Zona A` -> lo deposita en **`Zona B Inferior`** -> **introduce la pinza en el `Vaso D` para desinfectarse** -> vuelve a `HOME`.
3. **Por qué usamos 2 Políticas ACT separadas + Supervisor OpenCV (y NO un solo ACT mezclado):**
   - `ACT` puro no recibe instrucciones de texto en tiempo de inferencia. Si mezclamos episodios verdes y rojos en un mismo entrenamiento, el modelo promedia trayectorias (*mode averaging*) y duda sobre si ir o no al Vaso D.
   - Por tanto, grabamos **un único dataset de 64 episodios en LeLab** (para no perder tiempo cambiando configuraciones), pero **registramos exactamente qué índices son Verdes y cuáles Rojos** y entrenamos **2 modelos ACT independientes** filtrando con `--dataset.episodes="[...]"`:
     - **`act_sano`** (entrenado solo con los 32 episodios Verdes).
     - **`act_contaminado`** (entrenado solo con los 32 episodios Rojos).
   - Un **Supervisor OpenCV (`scripts/10_opencv_triage_supervisor.py`)** mira la cámara `top` cuando el brazo está en `HOME`, detecta cuántos cubos Verdes y Rojos quedan en `Zona A`, y ejecuta primero `act_sano` hasta vaciar los verdes de `Zona A`, y después `act_contaminado` hasta vaciar los rojos de `Zona A`.

---

## 2. TU PRIMERA MISIÓN (MODO INTERACTIVO PASO A PASO): GUIAR Y REGISTRAR LA GRABACIÓN HASTA 64 EPISODIOS

### Estado actual de la grabación (YA TENEMOS 24 EPISODIOS GRABADOS EN LELAB)
En nuestro dataset actual de LeLab ya hemos grabado **24 episodios (`0` al `23`)** con los 4 cubos (2 verdes y 2 rojos):
- **Episodios `0` a `7` (8 episodios):** Cubos **VERDES (`S`)** (`Zona A` -> `Zona B Superior` -> `HOME`).
- **Episodios `8` a `15` (8 episodios):** Cubos **ROJOS (`C`)** (`Zona A` -> `Zona B Inferior` -> `Vaso D` -> `HOME`).
- **Episodios `16` a `23` (8 episodios):** Cubos **VERDES (`S`)** (`Zona A` -> `Zona B Superior` -> `HOME`).

**Recuento actual:** `24 / 64` episodios (`16 Verdes` y `8 Rojos`). Faltan **40 episodios** (5 bloques de 8 episodios) para llegar a **64 episodios (`32 Verdes` + `32 Rojos`)**.

### Regla de interacción obligatoria conmigo durante la grabación
1. **No intentes avanzar al entrenamiento ni escribir todo el código de golpe mientras estamos grabando.**
2. **Al iniciar esta conversación:**
   - Localiza primero la carpeta exacta de nuestro dataset en `~/.cache/huggingface/lerobot/` (muéstrame qué carpeta es y verifica cuántos episodios tiene en disco ahora mismo).
   - Crea o actualiza inmediatamente un archivo de registro llamado **`NOTAS_EPISODIOS_64.md`** en la raíz del repositorio con la tabla de bloques, los índices exactos de episodios Verdes y Rojos, la configuración de la mesa en cada bloque y cualquier episodio defectuoso que haya que descartar.
   - Indícame claramente cuál es el **Bloque 4 (`24..31`)** que debo grabar ahora mismo y cómo debo colocar los cubos en la mesa antes de darle a grabar en LeLab.
3. **Cada vez que yo te escriba algo como `"ya grabé 8 episodios más de rojo"` (o te confirme que terminé un bloque):**
   - **Paso A:** Verifica en disco (en la carpeta del dataset en `~/.cache/huggingface/lerobot/...`) que los nuevos episodios se han guardado correctamente y comprueba el índice real del último episodio.
   - **Paso B:** Actualiza **`NOTAS_EPISODIOS_64.md`** añadiendo los índices exactos (ej. `24..31`), el color (`ROJO_C` o `VERDE_S`), la disposición de los cubos y las dos listas acumuladas listas para copiar en `--dataset.episodes`:
     - `EPISODIOS_VERDES = [...]`
     - `EPISODIOS_ROJOS = [...]`
     *(Si te digo que un episodio concreto salió mal, por ejemplo "el 27 salió mal", apúntalo en `EPISODIOS_DESCARTADOS` y exclúyelo de la lista de entrenamiento).*
   - **Paso C:** Respóndeme con un mensaje corto y directo:
     > `"✅ Guardados los episodios [X..Y] (Color). Progreso: N/64 (V verdes, R rojos). Ahora prepara la mesa así y graba el siguiente bloque de 8:"`
     seguido de la instrucción exacta del siguiente bloque.

### Plan Maestro de los 8 Bloques (64 Episodios) para cubrir todos los estados reales de la demo
> **Por qué variamos cuántos cubos quedan en `Zona A`:** En la demo en vivo, cuando el robot mueve el 1er cubo verde a `B_SUP`, para el 2º cubo verde solo queda **1 verde + 2 rojos en `Zona A`** (y 1 verde ya está en `B_SUP`). Luego, cuando pasa a los rojos, quedan **0 verdes en `Zona A`** (los 2 verdes ya están en `B_SUP`) y **2 rojos en `Zona A`**, y finalmente **1 solo rojo en `Zona A`**. Si entrenamos todos los episodios siempre con 4 cubos en `Zona A`, la red puede confundirse cuando queden 3, 2 o 1 cubo. Por eso distribuimos los 64 episodios así:

| Bloque | Índices | Cantidad y Color | Cómo colocar los cubos en la mesa antes de cada episodio | Estado |
| :---: | :---: | :--- | :--- | :--- |
| **1** | `0..7` | **8 Verdes (`S`)** | `Zona A`: 2 Verdes + 2 Rojos. Mover 1 Verde a `B_SUP` -> `HOME`. Reponer el Verde en `Zona A`. | ✅ **YA GRABADO** |
| **2** | `8..15` | **8 Rojos (`C` + Vaso D)** | `Zona A`: con cubos presentes. Mover 1 Rojo a `B_INF` -> desinfectar en `Vaso D` -> `HOME`. | ✅ **YA GRABADO** |
| **3** | `16..23` | **8 Verdes (`S`)** | `Zona A`: 2 Verdes + 2 Rojos (variando posiciones). Mover 1 Verde a `B_SUP` -> `HOME`. | ✅ **YA GRABADO** |
| **4** | `24..31` | **8 Rojos (`C` + Vaso D)** | **Escena de 1er Rojo en demo:** Deja los **2 Verdes ya aparcados en `B_SUP`** y pon los **2 Rojos en `Zona A`**. Mueve 1 Rojo a `B_INF` -> `Vaso D` -> `HOME`. Repón ese Rojo en `Zona A`. | ⏳ **SIGUIENTE A GRABAR** |
| **5** | `32..39` | **8 Verdes (`S`)** | **Escena de 2º Verde en demo:** Deja **1 Verde ya aparcado en `B_SUP`** y pon **1 Verde + 2 Rojos en `Zona A`**. Mueve el Verde restante a `B_SUP` -> `HOME`. Repón el Verde en `Zona A`. | ⏳ Pendiente |
| **6** | `40..47` | **8 Rojos (`C` + Vaso D)** | **Escena de último Rojo en demo:** Deja **2 Verdes en `B_SUP` + 1 Rojo en `B_INF`**, y pon **1 solo Rojo en `Zona A`**. Mueve ese último Rojo a `B_INF` -> `Vaso D` -> `HOME`. Repón ese Rojo en `Zona A`. | ⏳ Pendiente |
| **7** | `48..55` | **8 Verdes (`S`)** | **Refuerzo de bordes/posiciones difíciles:** 4 episodios con (2V+2R en `Zona A`) y 4 episodios con (1V+2R en `Zona A` y 1V en `B_SUP`), colocando el cubo verde objetivo en zonas variadas de `Zona A`. | ⏳ Pendiente |
| **8** | `56..63` | **8 Rojos (`C` + Vaso D)** | **Refuerzo de bordes/posiciones difíciles:** 4 episodios con (2R en `Zona A` y 2V en `B_SUP`) y 4 episodios con (1R en `Zona A`, 2V en `B_SUP`, 1R en `B_INF`), variando la posición del rojo en `Zona A`. | ⏳ Pendiente |

---

## 3. FASE 2 (CUANDO COMPLETEMOS LOS 64 EPISODIOS): AUDITORÍA Y ENTRENAMIENTO DE LAS 2 POLÍTICAS ACT

En cuanto yo te confirme el **Bloque 8 (`56..63`)**:
1. **Ejecuta la auditoría del dataset completo:**
   ```bash
   source scripts/env_hackathon.sh
   python scripts/02_audit_dataset.py \
     --repo-id <REPO_ID_DEL_DATASET> \
     --local-dir "<RUTA_LOCAL_DEL_DATASET>" \
     --output-json outputs/audit_64_episodes.json
   ```
2. **Filtra episodios defectuosos:**
   - Revisa `outputs/audit_64_episodes.json` y los episodios que hayamos marcado como malos en `NOTAS_EPISODIOS_64.md`.
   - Documenta en `NOTAS_EPISODIOS_64.md` exactamente cuántos episodios se grabaron, cuántos se descartaron y por qué (dato obligatorio para los **20 puntos de Datos** del jurado).
   - Genera las dos listas limpias definitivas: `EPISODIOS_VERDES_CLEAN` y `EPISODIOS_ROJOS_CLEAN`.
3. **Entrena las 2 políticas ACT en Apple Silicon (`mps`):**
   - **Política 1 (`act_sano` — solo episodios Verdes limpios):**
     ```bash
     lerobot-train \
       --dataset.repo_id=<REPO_ID_DEL_DATASET> \
       --dataset.root="<RUTA_LOCAL_DEL_DATASET>" \
       --dataset.episodes="<EPISODIOS_VERDES_CLEAN>" \
       --policy.type=act \
       --policy.device=mps \
       --policy.push_to_hub=false \
       --output_dir=outputs/train/act_sano \
       --job_name=act_sano \
       --steps=15000 \
       --batch_size=8 \
       --save_freq=5000
     ```
   - **Política 2 (`act_contaminado` — solo episodios Rojos limpios):**
     ```bash
     lerobot-train \
       --dataset.repo_id=<REPO_ID_DEL_DATASET> \
       --dataset.root="<RUTA_LOCAL_DEL_DATASET>" \
       --dataset.episodes="<EPISODIOS_ROJOS_CLEAN>" \
       --policy.type=act \
       --policy.device=mps \
       --policy.push_to_hub=false \
       --output_dir=outputs/train/act_contaminado \
       --job_name=act_contaminado \
       --steps=15000 \
       --batch_size=8 \
       --save_freq=5000
     ```

---

## 4. FASE 3 (MIENTRAS ENTRENA O AL TERMINAR): CREAR EL CALIBRADOR Y SUPERVISOR OPENCV (`scripts/10_opencv_triage_supervisor.py`)

Debes crear y dejar probado el script **`scripts/10_opencv_triage_supervisor.py`** con 3 modos (`--mode calibrate`, `--mode check`, `--mode run`):

### A) `--mode calibrate` (Calibrador interactivo de Zonas y Colores HSV)
- Abre la cámara `top` (usando el índice real de la cámara cenital en este Mac; verifica antes en el `info.json` del dataset de LeLab cómo se llaman las claves de cámara, p. ej. `io_robot_cam` / `io_robot_cam2` o `top` / `wrist`).
- Permite seleccionar o ajustar en pantalla las **ROIs (regiones de interés)**:
  1. `ZONA_A` (área izquierda donde aparecen los cubos `S` y `C`).
  2. `ZONA_B_SUP` (área derecha arriba de la línea horizontal: destino Sanos `S`).
  3. `ZONA_B_INF` (área derecha debajo de la línea horizontal: destino Contaminados `C`).
  4. `VASO_D` (posición del vaso descontaminante, para overlay visual en el HUD).
- Muestra barras deslizantes (*trackbars* de OpenCV) o ajuste interactivo para calibrar con la luz real de la sala:
  - Rango HSV para **Verde (`S` — Sano)**: `[H_min, S_min, V_min]` a `[H_max, S_max, V_max]`.
  - Rango HSV para **Rojo (`C` — Contaminado)**: doble rango HSV (`[0..10]` y `[165..180]`) + `S_min, V_min` para capturar bien el rojo bajo focos LED.
  - `min_area_px` y `max_area_px` para filtrar ruido y quedarnos solo con los cubos cuadrados.
- Al pulsar `s`, guarda toda la configuración en **`outputs/triage_vision_config.json`**.

### B) `--mode check` (HUD en vivo para verificar detección sin mover el robot)
- Carga `outputs/triage_vision_config.json`, abre la cámara `top` y dibuja en tiempo real:
  - Los recuadros de `ZONA_A`, `ZONA_B_SUP`, `ZONA_B_INF` y `VASO_D`.
  - Bounding boxes verdes sobre cada cubo `S` detectado (con su centroide) y rojas sobre cada cubo `C` detectado.
  - Un panel HUD arriba para el jurado que muestre:
    - `ZONA A: X Sanos (S) | Y Contaminados (C)`
    - `ZONA B SUP (Sanos): X | ZONA B INF (Desechos): Y`
    - `SIGUIENTE ORDEN: [EVACUAR SANO -> act_sano] / [CUARENTENA + VASO D -> act_contaminado] / [ZONA A LIMPIA - COMPLETADO]`

### C) `--mode run` (Orquestador Autónomo en Bucle Cerrado para la Demo 5/5)
- Con el brazo inicialmente en `HOME`:
  1. **Escaneo:** Abre brevemente la cámara `top` (lee ~15 frames para estabilizar auto-exposición), cuenta cuántos cubos **Verdes (`S`)** y **Rojos (`C`)** hay en `ZONA_A`, muestra/guarda un snapshot del HUD y **cierra `cv2.VideoCapture` antes de lanzar `lerobot-rollout`** (¡crítico en macOS para que `lerobot-rollout` pueda abrir la cámara `top` sin conflicto de dispositivo ocupado!).
  2. **Decisión por prioridad de Bioseguridad (`Clean-First`):**
     - **Si `verdes_en_A > 0`:**
       - Lanza `lerobot-rollout` con `--policy.path=outputs/train/act_sano/checkpoints/last/pretrained_model`, `--device=mps`, `--robot.max_relative_target=15`, `--duration=<DURACION_SANO>` (ej. `15` s) y el diccionario `--robot.cameras` con los nombres exactos de cámara con los que se grabó el dataset en LeLab.
     - **Si `verdes_en_A == 0` y `rojos_en_A > 0`:**
       - Lanza `lerobot-rollout` con `--policy.path=outputs/train/act_contaminado/checkpoints/last/pretrained_model`, `--device=mps`, `--robot.max_relative_target=15`, `--duration=<DURACION_CONTAMINADO>` (ej. `18` s, incluye paso por el Vaso D).
     - **Si `verdes_en_A == 0` y `rojos_en_A == 0`:**
       - Declara **`MISIÓN COMPLETADA: ZONA A DESPEJADA Y CLASIFICADA SIN CONTAMINACIÓN CRUZADA`** y termina.
  3. **Verificación en bucle cerrado + Auto-Reintento:**
     - Al terminar cada `lerobot-rollout` (cuando el brazo ya volvió a `HOME`), vuelve a abrir la cámara `top` y comprueba si el número de cubos del color objetivo en `ZONA_A` disminuyó.
     - Si el cubo sigue en `ZONA_A` (porque resbaló), registra `[AUTO-RECOVERY] Reintentando...` y vuelve a ejecutar la política (hasta `--max-retries 2`) sin intervención humana.

---

## 5. EMPECEMOS AHORA MISMO
Por favor, **empieza ejecutando únicamente el inicio de la Fase 1**:
1. Busca en `~/.cache/huggingface/lerobot/` nuestro dataset actual de 24 episodios y confírmame su nombre y que ves los 24 episodios (`0..23`).
2. Crea el archivo `NOTAS_EPISODIOS_64.md` registrando los Bloques 1 (`0..7` verdes), 2 (`8..15` rojos) y 3 (`16..23` verdes).
3. Pídeme que coloque la mesa para el **Bloque 4 (`24..31` — 8 episodios Rojos con los 2 Verdes ya en `Zona B Superior`)** y espera mi mensaje `"ya grabé 8 episodios más de rojo"`.
```
