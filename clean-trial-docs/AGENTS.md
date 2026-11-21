# AGENTS.md — Directrices para el agente de IA

Lee este archivo entero antes de escribir nada. Después lee README.md, MEMORY.md y COMO_FUNCIONA.md.

## Contexto
Hackathon Bilbao II (20–22 nov 2026). Hay un brazo robótico SO-101 (Leader + Follower) conectado a una Mac con 2 cámaras. El entrenamiento, la calibración, las demostraciones y la ejecución del modelo se hacen en esa Mac con LeLab / LeRobot. Los brazos ya están calibrados.

## Objetivo
Construir un panel web remoto, muy sencillo, que funcione como un mando: lanzar la tarea que el robot ya sabe hacer, pararla y ver su estado.

## Regla de oro
La percepción y la lógica de decisión viven en el modelo entrenado de la Mac. El panel:
- NO hace visión.
- NO abre cámaras.
- NO abre puertos serie.
- NO contiene lógica de decisión.

El panel solo lanza y para un comando externo, y muestra en pantalla lo que está pasando.

## Alcance: tres controles
1. **Iniciar tarea**
2. **Parar tarea** (parada de emergencia)
3. **Ver estado de la tarea**: libre / ejecutando / detenido / error

Extra opcional, solo si sobra tiempo: un log corto de lo que ocurre.

El panel es opcional. No debe interferir con LeLab. Si los jueces lo rechazan, el robot se lanza desde LeLab sin el panel. No hace falta modo monitor.

## Stack (no cambiar)
- Backend: Python con FastAPI + Uvicorn + PyYAML.
- Frontend: un único `index.html` simple, sin build y sin frameworks.
- Acceso: `http://localhost:8000` o `http://<IP_MAC>:8000` desde la red local.

## Requisitos técnicos
- **Configuración en archivo** (`config.example.yaml`, y el real `config.yaml` sin subir a Git): comando de la tarea, puerto y demás ajustes. Nada de rutas ni puertos escritos a mano en el código.
- **`mock_mode: true` por defecto.** En ese modo la tarea se simula y no toca el robot. Pasar a `false` solo para usar los comandos reales.
- **La tarea se ejecuta como subproceso** en su propio grupo de procesos.
- **STOP** mata el subproceso y todo su grupo con `os.killpg`.
- **Exclusión mutua** con `threading.Lock`: solo una tarea a la vez. Si se pulsa iniciar mientras ya hay una, se rechaza con un mensaje claro.
- **Estado en tiempo real:** el frontend lo consulta cada poco tiempo.
- **Seguridad básica:** el comando a ejecutar sale solo del archivo de configuración, nunca de lo que envíe el navegador.
- **Errores controlados:** el panel nunca debe quedarse colgado ni caerse por un fallo de la tarea.

## Estructura del repo
- `AGENTS.md` — este archivo
- `MEMORY.md` — bitácora de avances y pendientes
- `README.md` — qué es y cómo arrancarlo
- `docs/COMO_FUNCIONA.md` — explicación corta y muy descriptiva
- `config.example.yaml`
- `runner.py` — lanza, para y vigila la tarea
- `main.py` — servidor FastAPI
- `index.html` — el panel

## Forma de trabajar
1. Primero la base documental, después el código.
2. Código simple y legible, con comentarios en español.
3. Pasos pequeños. No añadir dependencias ni funciones fuera del alcance.
4. Probar siempre en simulación (`mock_mode: true`) antes de pasar a comandos reales.
5. Al terminar cada paso, actualizar `MEMORY.md`.
6. Al terminar cada archivo, explicar en pocas líneas qué hace y por qué está hecho así.
