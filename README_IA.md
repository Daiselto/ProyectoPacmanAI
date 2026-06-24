# 🎮 Proyecto Pac-Man IA — Sistemas Inteligentes
**Universidad Tecnológica Centroamericana (UNITEC)**  
Clase: Sistemas Inteligentes  
Estudiante: Daniel Elvir

---

## 📋 Descripción

Proyecto basado en el juego clásico de Pac-Man implementado en Python con Pygame, al cual se le agregó inteligencia artificial a los 4 fantasmas usando algoritmos de búsqueda, comportamientos diferenciados y aprendizaje por refuerzo con Stable-Baselines3 y Gymnasium.

---

## 🧠 Comportamientos Implementados

| Fantasma | Color | Comportamiento | Técnica |
|---|---|---|---|
| **Blinky** | 🔴 Rojo | Persecución directa con aprendizaje | PPO (Stable-Baselines3) + A* |
| **Pinky** | 🩷 Rosa | Predicción 3 casillas adelante de Pac-Man | A* con offset de dirección |
| **Inky** | 🩵 Celeste | Coordinación con Blinky para acorralar | Vector Blinky → Pivot × 2 |
| **Clyde** | 🟠 Naranja | Movimiento aleatorio (el tonto) | Random |

### Comportamientos adicionales
- **Estado vulnerable**: cuando Pac-Man come un power pellet, todos los fantasmas huyen en dirección opuesta
- **Spectacles**: cuando un fantasma es comido, vuelve corriendo a su punto de spawn
- **Aprendizaje por refuerzo**: Blinky utiliza un modelo PPO entrenado con Gymnasium

---

## 🤖 Inteligencia Artificial

### Algoritmo A*
Implementado en `src/utils/path_finder.py`. Utilizado por todos los fantasmas para calcular rutas óptimas dentro del laberinto.

### Entorno Gymnasium (`src/env/ghost_env.py`)
Entorno personalizado donde los 4 fantasmas son el agente y Pac-Man se mueve aleatoriamente durante el entrenamiento.

- **Observación**: posición de los 4 fantasmas + posición de Pac-Man (10 valores)
- **Acciones**: 256 combinaciones (4 direcciones × 4 fantasmas en base 4)
- **Recompensas**:
  - `+1.0` por acercarse a Pac-Man
  - `-0.5` por alejarse
  - `+10.0` por atrapar a Pac-Man
  - `+0.5 × n` bonus por coordinación (n fantasmas cerca simultáneamente)

### Entrenamiento PPO (`train_ghosts.py`)
Algoritmo Proximal Policy Optimization de Stable-Baselines3, entrenado durante 30 episodios.

---

## 📊 Métricas

Al finalizar cada partida se registran automáticamente métricas en consola y en `metricas_partidas.csv`:

| Métrica | Descripción |
|---|---|
| Capturas totales | Veces que los fantasmas atraparon a Pac-Man |
| Capturas por fantasma | Desglose por Blinky, Pinky, Inky, Clyde |
| Power pellets comidos | Veces que Pac-Man activó el modo vulnerable |
| Fantasmas comidos | Fantasmas eliminados por Pac-Man |
| Duración (frames) | Duración total de la partida |

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.12**
- **Pygame 2.6** — motor del juego
- **Gymnasium** — entorno de entrenamiento para IA
- **Stable-Baselines3** — algoritmo PPO para aprendizaje por refuerzo
- **TensorFlow / Keras** — backend de redes neuronales
- **NumPy** — operaciones matriciales

---

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install pygame pygame-menu numpy gymnasium stable-baselines3 tensorflow tensorboard tqdm rich
```

### Correr el juego
```bash
python main.py --layout classic
```

### Entrenar el modelo de fantasmas (30 episodios)
```bash
python train_ghosts.py
```

El modelo se guarda automáticamente en `models/ghosts_ppo_final.zip` y se carga al iniciar el juego.

### Ver métricas acumuladas
```bash
type metricas_partidas.csv   # Windows
cat metricas_partidas.csv    # Mac/Linux
```

---

## 📁 Estructura del Proyecto

```
py-pacman/
├── main.py                        # Punto de entrada del juego
├── train_ghosts.py                # Script de entrenamiento PPO
├── metricas_partidas.csv          # Historial de métricas por partida
├── models/
│   └── ghosts_ppo_final.zip       # Modelo PPO entrenado
├── src/
│   ├── game.py                    # Loop principal del juego
│   ├── ghost.py                   # Lógica e IA de los fantasmas
│   ├── pacman.py                  # Lógica del jugador
│   ├── map.py                     # Laberinto
│   ├── metrics.py                 # Sistema de métricas
│   ├── constants.py               # Configuraciones
│   └── env/
│       ├── ghost_env.py           # Entorno Gymnasium para fantasmas
│       ├── pacman_env.py          # Entorno Gymnasium para Pac-Man
│       ├── trained_agent.py       # Carga y usa el modelo PPO
│       └── agent.py               # Clase abstracta de agente
└── res/                           # Sprites y recursos del juego
```

---

## 📈 Resultados del Entrenamiento

El modelo fue entrenado durante 30 episodios. Las recompensas aumentaron progresivamente, mostrando que los fantasmas aprendieron a coordinarse y acercarse a Pac-Man de forma más efectiva con el tiempo.

```
Episodio  1/30 | Recompensa:    74.50
Episodio  5/30 | Recompensa:   166.50
Episodio 10/30 | Recompensa:   188.50
Episodio 20/30 | Recompensa:   173.50
Episodio 30/30 | Recompensa:   251.00
```

---

## 📝 Créditos

- Proyecto base: [paolodelia99/py-pacman](https://github.com/paolodelia99/py-pacman)
- Modificaciones e IA: Daniel Elvir — UNITEC 2026
