🎮 Proyecto Pac-Man IA — Sistemas Inteligentes

Universidad Tecnológica Centroamericana (UNITEC)

Clase: Sistemas Inteligentes

Estudiante: Daniel Elvir


📋 Descripción

Proyecto basado en el juego clásico de Pac-Man implementado en Python con Pygame, al cual se le agregó inteligencia artificial a los 4 fantasmas usando algoritmos de búsqueda, comportamientos diferenciados y aprendizaje por refuerzo con Stable-Baselines3 y Gymnasium. Cada fantasma tiene su propio modelo PPO entrenado individualmente con recompensas diferenciadas según su rol.


🧠 Comportamientos Implementados

FantasmaColorComportamientoTécnicaBlinky🔴 RojoPersecución directa — sigue la espalda de Pac-ManPPO individual + A*Pinky🩷 RosaPredicción — se posiciona 3 casillas adelante de Pac-ManPPO individual + A*Inky🩵 CelesteCoordinación con Blinky para acorralar a Pac-ManPPO individual + vector Blinky→Pivot×2Clyde🟠 NaranjaImpredecible — movimiento con recompensas aleatoriasPPO individual + random

Comportamientos adicionales


Estado vulnerable: cuando Pac-Man come un power pellet, todos los fantasmas huyen en dirección opuesta
Spectacles: cuando un fantasma es comido, regresa corriendo a su punto de spawn
Aprendizaje individual: cada fantasma tiene su propio modelo PPO entrenado con recompensas específicas a su rol



🤖 Inteligencia Artificial

Algoritmo A*

Implementado en src/utils/path_finder.py. Utilizado por todos los fantasmas para calcular rutas óptimas dentro del laberinto evitando paredes.

Entornos Gymnasium (src/env/ghost_env_individual.py)

Se crearon 4 entornos de entrenamiento individuales, uno por fantasma. En cada uno Pac-Man se mueve aleatoriamente y el fantasma aprende a cumplir su objetivo específico.

EntornoObservaciónRecompensaBlinkyEnvPosición ghost + player + diferencia+1 acercarse, +15 atraparPinkyEnvPosición ghost + player + velocidad + target adelante+1 estar cerca del punto adelante, +10 llegarInkyEnvPosición inky + blinky + player + target coordinado+1 llegar al punto de coordinación, +10 llegarClydeEnvPosición ghost + playerRecompensa aleatoria (-0.5 a +0.5)

Entrenamiento PPO (train_ghosts_individual.py)

Algoritmo Proximal Policy Optimization de Stable-Baselines3. Cada fantasma se entrena por separado durante 30 episodios con su entorno y recompensas propias.

==================================================
  Entrenamiento Individual de Fantasmas con PPO
  Episodios por fantasma: 30
==================================================
  [Blinky] Episodio  1/30 | Recompensa:   -92.50
  [Blinky] Episodio 30/30 | Recompensa:   -91.50
  ✅ Blinky guardado en models/blinky_ppo.zip

  [Pinky]  Episodio  1/30 | Recompensa:    45.00
  [Pinky]  Episodio 30/30 | Recompensa:    87.50
  ✅ Pinky guardado en models/pinky_ppo.zip

  [Inky]   Episodio  1/30 | Recompensa:    38.00
  [Inky]   Episodio 30/30 | Recompensa:    92.00
  ✅ Inky guardado en models/inky_ppo.zip

  [Clyde]  Episodio  1/30 | Recompensa:     3.37
  [Clyde]  Episodio 30/30 | Recompensa:     2.14
  ✅ Clyde guardado en models/clyde_ppo.zip


📊 Métricas

Al finalizar cada partida se registran automáticamente métricas en consola y en metricas_partidas.csv:

MétricaDescripciónCapturas totalesVeces que los fantasmas atraparon a Pac-ManCapturas por fantasmaDesglose por Blinky, Pinky, Inky, ClydePower pellets comidosVeces que Pac-Man activó el modo vulnerableFantasmas comidosFantasmas eliminados por Pac-ManDuración (frames)Duración total de la partida

El CSV permite comparar el desempeño de la IA entre partidas y ver cómo las métricas evolucionan con el tiempo.


🛠️ Tecnologías Utilizadas


Python 3.12
Pygame 2.6 — motor del juego
Gymnasium — entornos de entrenamiento para IA
Stable-Baselines3 — algoritmo PPO para aprendizaje por refuerzo
TensorFlow / Keras — backend de redes neuronales
NumPy — operaciones matriciales



🚀 Instalación y Uso

Requisitos

bashpip install pygame pygame-menu numpy gymnasium stable-baselines3 tensorflow tensorboard tqdm rich

Correr el juego

bashpython main.py --layout classic

Entrenar los 4 modelos individuales (30 episodios cada uno)

bashpython train_ghosts_individual.py

Los modelos se guardan automáticamente en la carpeta models/ y se cargan al iniciar el juego.

Ver métricas acumuladas

bashtype metricas_partidas.csv   # Windows
cat metricas_partidas.csv    # Mac/Linux


📁 Estructura del Proyecto

py-pacman/
├── main.py                            # Punto de entrada del juego
├── train_ghosts_individual.py         # Entrenamiento PPO individual por fantasma
├── train_ghosts.py                    # Entrenamiento PPO conjunto (referencia)
├── metricas_partidas.csv              # Historial de métricas por partida
├── models/
│   ├── blinky_ppo.zip                 # Modelo PPO de Blinky
│   ├── pinky_ppo.zip                  # Modelo PPO de Pinky
│   ├── inky_ppo.zip                   # Modelo PPO de Inky
│   └── clyde_ppo.zip                  # Modelo PPO de Clyde
├── src/
│   ├── game.py                        # Loop principal del juego
│   ├── ghost.py                       # Lógica e IA de los fantasmas
│   ├── pacman.py                      # Lógica del jugador
│   ├── map.py                         # Laberinto
│   ├── metrics.py                     # Sistema de métricas y CSV
│   ├── constants.py                   # Configuraciones
│   └── env/
│       ├── ghost_env.py               # Entorno Gymnasium conjunto
│       ├── ghost_env_individual.py    # Entornos Gymnasium individuales
│       ├── pacman_env.py              # Entorno Gymnasium para Pac-Man
│       ├── trained_agent.py           # Carga y usa los 4 modelos PPO
│       └── agent.py                   # Clase abstracta de agente
└── res/                               # Sprites y recursos del juego


📝 Créditos


Proyecto base: paolodelia99/py-pacman
Modificaciones e IA: Daniel Elvir — UNITEC 2026