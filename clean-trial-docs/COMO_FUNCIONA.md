# Cómo funciona

## La idea en una frase
El panel es como el mando de una tele: no sabe hacer la película, solo dice "empieza", "para" y "qué está pasando".

## Las piezas
1. **El robot y su modelo (en la Mac).** Es quien ve con las cámaras y decide qué hacer. Todo eso ya vive en LeLab / LeRobot.
2. **El servidor del panel (Python).** Corre en la misma Mac. Cuando recibe "iniciar", lanza el comando del modelo como un proceso aparte. Cuando recibe "parar", lo detiene.
3. **La página (`index.html`).** Se abre en el navegador, muestra tres controles y pregunta al servidor cada poco tiempo cómo va la tarea.

## El recorrido de una orden
1. Alguien pulsa **Iniciar** en la página.
2. La página avisa al servidor.
3. El servidor comprueba que no hay otra tarea en marcha. Si la hay, responde que está ocupado.
4. Si está libre, lanza el comando que está escrito en el archivo de configuración y pasa a estado **ejecutando**.
5. La página pregunta el estado cada poco y lo muestra: libre, ejecutando, detenido o error.
6. Si se pulsa **Parar**, el servidor detiene el proceso y también todo lo que haya arrancado ese proceso, para que nada quede vivo. Pasa a **detenido**.

## Por qué está hecho así
- **El comando viene del archivo de configuración**, no de la página. Así nadie puede mandar al servidor ejecutar algo distinto desde el navegador.
- **Solo una tarea a la vez.** Evita que dos órdenes se pisen y el robot reciba instrucciones mezcladas.
- **Modo simulación por defecto.** Se puede probar todo sin tocar el robot. Cuando funciona, se cambia un ajuste y pasa a usar los comandos reales.
- **El panel no ve ni decide nada.** Si falla o se rechaza, el robot se lanza desde LeLab igual que siempre.

## Estados posibles
- **Libre:** no hay ninguna tarea en marcha.
- **Ejecutando:** la tarea está en marcha.
- **Detenido:** se paró con el botón de parar.
- **Error:** la tarea terminó mal o no pudo arrancar.
