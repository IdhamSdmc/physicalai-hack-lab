# Hackathon LeRobot Bilbao II (Physical AI Hackathon Bilbao)

**Lugar:** La Perrera, Bilbao
**Fechas:** viernes 20 de noviembre de 2026 a las 16:00 hasta el domingo 22 a las 14:00
**Robot:** brazo SO-101 (líder para teleoperar y follower de 6 servos), con LeRobot v0.7.0
**Fuentes:** [web del evento](https://hackathonbilbao.eus/home-page-eus) · [guía SO-101](https://hackathonbilbao.eus/guia-so-101)

## Premios

| Puesto | Premio |
| --- | --- |
| 1.º | 1.500 € |
| 2.º | 900 € |
| 3.º | 600 € |

## Horario del domingo 22

| Hora | Qué pasa |
| --- | --- |
| 9:00 | Cierre de entregas |
| 11:30 a 12:30 | Presentaciones |
| 12:30 | Deliberación del jurado |
| 13:00 a 13:30 | Entrega de premios |

## Qué se entrega

1. **Un vídeo de 1 minuto** que muestre el proyecto.
2. **Una demo en vivo de 3 minutos** para explicar el proyecto al jurado.

## Baremo: 100 puntos y hasta 5 de bonus

La nota final es la media de los jurados.

| Criterio | Qué se valora | Puntos |
| --- | --- | --- |
| Demo en vivo | Cinco intentos seguidos de la tarea completa sin tocar el robot. Cada éxito suma 6 puntos. | 30 |
| Datos y entrenamiento | Cantidad, variedad y consistencia de las demostraciones, las cámaras y el encuadre, y cómo habéis iterado sobre los fallos. | 20 |
| Dificultad de la tarea | Ambición realista para 48 horas, desde coger un objeto hasta tareas de varios pasos que toleran cambios de posición o de luz. | 15 |
| Utilidad e impacto | Si la solución resuelve un problema real. | 15 |
| Innovación e ingenio | Originalidad del enfoque y de la solución. | 10 |
| Presentación y equipo | Cómo contáis lo que habéis hecho y cómo respondéis a las preguntas del jurado. | 10 |
| **Bonus** | Dataset publicado en Hugging Face (+3) y repositorio reproducible (+2). | **+5** |

## Qué preparar para el domingo

- **La demo:** el robot listo para cinco intentos seguidos. Un fallo ajeno al modelo, como un cable o una parada de emergencia, se puede repetir una vez, a criterio del jurado.
- **Las decisiones del equipo:** el jurado preguntará cuántos episodios grabasteis, cuántos descartasteis y qué cambiasteis tras el primer entrenamiento.
- **El dataset en Hugging Face:** +3 puntos.
- **Un repositorio reproducible,** con código y pasos para repetir el resultado: +2 puntos.

## Consejos clave de la guía

- Unos **50 episodios** para coger y dejar un objeto, unos **10 por cada posición** del objeto. **3 de 5 aciertos ya es una buena demo.**
- Grabar lleva aproximadamente **1 hora por cada 20 episodios**.
- Usad cámaras fijas (con la misma configuración al grabar y al ejecutar), agarrad siempre igual, mantened la luz estable y empezad desde la misma postura.
- Política recomendada: **ACT**. Entrenad pronto, porque tarda varias horas, y en el evento no hay GPU (portátil con NVIDIA, Colab o HF Jobs).
- Ciclo de trabajo: probar, mirar dónde falla y grabar más episodios de esos casos.
- Si algo no se arregla, avisad a un mentor de Bravo Robots, escribid en el grupo de WhatsApp o preguntad en el Discord de LeRobot.
