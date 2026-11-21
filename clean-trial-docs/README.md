# Panel de control web para el brazo SO-101

Hackathon Bilbao II · 20–22 nov 2026 · La Perrera Espazioa

## Qué es
Un mando web muy sencillo para un brazo robótico SO-101. Tiene tres controles:
- Iniciar la tarea
- Parar la tarea (parada de emergencia)
- Ver el estado de la tarea

El panel no controla el brazo directamente. El robot y su modelo entrenado funcionan en la Mac con LeLab / LeRobot. El panel solo lanza y para ese proceso, y muestra su estado.

## Cómo se usa (cuando esté construido)
1. Copiar `config.example.yaml` a `config.yaml` y ajustar el comando de la tarea.
2. Arrancar el servidor en la Mac.
3. Abrir `http://localhost:8000` en la Mac, o `http://<IP_MAC>:8000` desde otro dispositivo de la misma red.
4. Probar primero con `mock_mode: true` (simulación, no toca el robot).
5. Cuando todo funcione, cambiar a `mock_mode: false` para usar los comandos reales.

## Importante
- El panel es opcional. Si no se usa, el robot se lanza desde LeLab como siempre.
- No hace visión, no abre cámaras ni puertos serie y no toma decisiones.

## Stack
Python (FastAPI + Uvicorn + PyYAML) y un `index.html` simple sin build.

## Documentación
- `AGENTS.md` — reglas para el agente de IA
- `MEMORY.md` — avances y pendientes
- `docs/COMO_FUNCIONA.md` — cómo encajan las piezas
