# MEMORY.md — Bitácora del proyecto

Actualizar al terminar cada paso. Mantenerlo corto.

## Decisiones tomadas
- El panel es un mando: iniciar tarea, parar tarea, ver estado.
- La tarea asignada por defecto es la que el robot ya sabe hacer; el panel solo la lanza.
- El panel es opcional y no interfiere con LeLab. No hay modo monitor.
- Stack: FastAPI + Uvicorn + PyYAML y un `index.html` sin build.
- `mock_mode: true` por defecto.
- Los brazos ya están calibrados; el entrenamiento se hace en la Mac.

## Hecho
- [x] Definido el alcance y las reglas (`AGENTS.md`).
- [x] README y explicación corta preparados.

## Pendiente
- [ ] `config.example.yaml`
- [ ] `runner.py` (lanzar, parar y vigilar el subproceso)
- [ ] `main.py` (servidor y endpoints de iniciar, parar y estado)
- [ ] `index.html` (los tres controles y el estado en pantalla)
- [ ] Probar todo en simulación
- [ ] Pasar a comandos reales (`mock_mode: false`) y probar con el robot

## Notas
(Aquí apunta lo que vaya saliendo: errores, dudas, cambios de plan.)
