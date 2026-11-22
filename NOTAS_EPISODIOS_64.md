# Registro de Grabación (64 Episodios) — Zero-Cross-Contamination Triage

## Información del Dataset (LeLab)
- **Repo ID:** `local/clasificador_objetos_20261121_171107`
- **Ruta local:** `~/.cache/huggingface/lerobot/local/clasificador_objetos_20261121_171107`
- **FPS / Robot:** `30` FPS | `so_follower`
- **Claves de cámara en `info.json`:**
  - `observation.images.camera_top` (`480x640x3`) — Cámara cenital
  - `observation.images.camera_robot` (`480x640x3`) — Cámara de muñeca/pinza
- **Progreso total en disco:** `64 / 64` episodios (`0..63`) | `28,519` frames ✅ **COMPLETADO**

---

## Tabla de Bloques (8 Bloques x 8 Episodios = 64)

| Bloque | Índices | Cantidad y Color | Configuración de la mesa antes de cada episodio | Estado |
| :---: | :---: | :--- | :--- | :--- |
| **1** | `0..7` | **8 Verdes (`VERDE_S`)** | `Zona A`: 2 Verdes + 2 Rojos. Mover 1 Verde a `B_SUP` -> `HOME`. Reponer Verde en `Zona A`. Task: `pruebas iniciales` (`376..449` frames). | ✅ Grabado |
| **2** | `8..15` | **8 Rojos (`ROJO_C` + Vaso D)** | `Zona A`: cubos presentes. Mover 1 Rojo a `B_INF` -> desinfectar en `Vaso D` -> `HOME`. Task: `objetos contaminados pruebas iniciales` (`493..567` frames). | ✅ Grabado |
| **3** | `16..23` | **8 Verdes (`VERDE_S`)** | `Zona A`: 2 Verdes + 2 Rojos (variando posiciones). Mover 1 Verde a `B_SUP` -> `HOME`. Task: `objetos sanos mas pruebas` (**⚠️ Ep. 18 tiene solo `29` frames / <1s**; resto `328..435` frames). | ✅ Grabado *(1 corto: ep. 18)* |
| **4** | `24..31` | **8 Rojos (`ROJO_C` + Vaso D)** | **Nombre en LeLab:** `b4_rojo_2R_en_A_2V_en_Bsup`<br>**Escena de 1er Rojo en demo:** Dejar los **2 Verdes aparcados en `B_SUP`** y poner **2 Rojos en `Zona A`**. Mover 1 Rojo a `B_INF` -> `Vaso D` -> `HOME`. Reponer Rojo en `Zona A` (`480..556` frames). | ✅ Grabado |
| **5** | `32..39` | **8 Verdes (`VERDE_S`)** | **Nombre en LeLab:** `b5_verde_1V_2R_en_A_1V_en_Bsup`<br>**Escena de 2º Verde en demo:** Dejar **1 Verde aparcado en `B_SUP`** y poner **1 Verde + 2 Rojos en `Zona A`**. Mover el Verde restante a `B_SUP` -> `HOME`. Reponer Verde en `Zona A` (`333..446` frames). | ✅ Grabado |
| **6** | `40..47` | **8 Rojos (`ROJO_C` + Vaso D)** | **Nombre en LeLab:** `b6_rojo_1R_en_A_2V_Bsup_1R_Binf`<br>**Escena de último Rojo en demo:** Dejar **2 Verdes en `B_SUP` + 1 Rojo en `B_INF`**, y poner **1 solo Rojo en `Zona A`**. Mover ese último Rojo a `B_INF` -> `Vaso D` -> `HOME`. Reponer Rojo en `Zona A` (`400..501` frames). | ✅ Grabado |
| **7** | `48..55` | **8 Verdes (`VERDE_S`)** | **Nombre en LeLab:** `b7_verde_bordes_dificiles`<br>**Refuerzo de bordes/posiciones difíciles:** 4 episodios con (2V+2R en `Zona A`) y 4 con (1V+2R en `Zona A` y 1V en `B_SUP`), colocando el cubo verde objetivo en bordes/zonas variadas (`367..447` frames). | ✅ Grabado |
| **8** | `56..63` | **8 Rojos (`ROJO_C` + Vaso D)** | **Nombre en LeLab:** `b8_rojo_bordes_dificiles`<br>**Refuerzo de bordes/posiciones difíciles:** 4 episodios con (2R en `Zona A` y 2V en `B_SUP`) y 4 con (1R en `Zona A`, 2V en `B_SUP`, 1R en `B_INF`), variando la posición del rojo (`470..566` frames). | ✅ Grabado |

---

## Historial de Limpiezas
- **Limpieza 1 (Bloque 4 fallido):** Eliminados los episodios `24..31` (`objetos rojos 8 mas`) mediante `lerobot_edit_dataset --operation.type delete_episodes`, restaurando el dataset a `24` episodios (`0..23`, `10,181` frames). Copia de respaldo movida a `outputs/backups/clasificador_objetos_20261121_171107_old_24_31`.

---

## Auditoría Final (Fase 2 — `outputs/audit_64_episodes.json`)

- **Episodios grabados en el dataset final:** `64` (`32` Verdes + `32` Rojos, `28,519` frames, duración media `14.82 s`). *(Más 8 fallidos regrabados en Bloque 4).*
- **Episodios limpios tras auditoría:** **`60` (`93.8%`)** — **`30` Verdes limpios + `30` Rojos limpios**.
- **Episodios descartados (`4` / `6.2%`):**
  - **`Episodio 7` (Verde):** Tiempo muerto inicial de `1.6 s` quieto al arrancar (umbral > `1.5 s`).
  - **`Episodio 11` (Rojo):** Tiempo muerto inicial de `1.7 s` quieto al arrancar.
  - **`Episodio 15` (Rojo):** Tiempo muerto inicial de `1.8 s` quieto al arrancar.
  - **`Episodio 18` (Verde):** Duración anómala de `0.93 s` (`29` frames), `100%` del tiempo inmóvil (*hesitation*) + alto *jerk* P98.

```python
EPISODIOS_VERDES_CLEAN = [0, 1, 2, 3, 4, 5, 6, 16, 17, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55]  # 30 limpios
EPISODIOS_ROJOS_CLEAN = [8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63]  # 30 limpios
EPISODIOS_DESCARTADOS = [7, 11, 15, 18]  # 4 descartados (6.2%)
```

## Datasets Limpios y Modelos en Hugging Face Hub (Entrenamiento Dual en L40S)
- **Verde (`act_sano` v1):**
  - Dataset: [`coalva94/bilbao_triage_sano`](https://huggingface.co/datasets/coalva94/bilbao_triage_sano) (`30` episodios, `11,887` frames)
  - Job HF: `6b01e587efaf5e14f4b62b68` | Modelo: `coalva94/act_2026-11-21_18-40-05` (`20,000` steps)
- **Rojo (`act_contaminado` v1):**
  - Dataset: [`coalva94/bilbao_triage_contaminado`](https://huggingface.co/datasets/coalva94/bilbao_triage_contaminado) (`30` episodios, `15,098` frames)
  - Job HF: `6b01e60d4fdc5ab3c55e0ff9` | Modelo: `coalva94/act_2026-11-21_18-42-20` (`20,000` steps)

---

## Refuerzo V2 (`88` Episodios Totales -> `42 + 42` Limpios)
- **Nuevos episodios grabados (`24`):**
  - `64..75` (`12` eps): `re-train-verdes` (`13.0s–14.6s`, `0` anomalías en `outputs/audit_88_episodes.json`)
  - `76..87` (`12` eps): `re-train-rojos` (`15.7s–21.6s`, `0` anomalías en `outputs/audit_88_episodes.json`)
- **Datasets V2 en Hugging Face Hub:**
  - `coalva94/bilbao_triage_sano_v2` (`42` episodios limpios, `16,877` frames)
  - `coalva94/bilbao_triage_contaminado_v2` (`42` episodios limpios, `21,559` frames)
- **Jobs Fine-Tuning V2 en `1x Nvidia L40S` (`12,000` steps, `--timeout 2h`, `image_transforms.backend=gpu`):**
  - **Verde (`act_sano_v2`):** Job `6b0214434fdc5ab3c55e5a1d` | Base `coalva94/act_2026-11-21_18-40-05` -> [`coalva94/act_sano_v2`](https://huggingface.co/coalva94/act_sano_v2) *(reemplaza los previos timeout `6b020c8defaf5e14f4b6726f`)*
  - **Rojo (`act_contaminado_v2`):** Job `6b0214434fdc5ab3c55e5a1f` | Base `coalva94/act_2026-11-21_18-42-20` -> [`coalva94/act_contaminado_v2`](https://huggingface.co/coalva94/act_contaminado_v2) *(reemplaza los previos timeout `6b020c8defaf5e14f4b67273`)*
