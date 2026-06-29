import numpy as np
import os
import pickle
from stable_baselines3 import PPO


class GhostLearner:
    """
    Sistema de aprendizaje continuo para los fantasmas.
    Recolecta experiencias de partidas reales y las guarda
    para entrenar con train_ghosts_individual.py.
    """

    NOMBRES = {0: "Blinky", 1: "Pinky", 2: "Inky", 3: "Clyde"}
    MODEL_PATHS = {
        0: "models/blinky_ppo.zip",
        1: "models/pinky_ppo.zip",
        2: "models/inky_ppo.zip",
        3: "models/clyde_ppo.zip",
    }
    EXPERIENCE_PATHS = {
        0: "models/blinky_experiences.pkl",
        1: "models/pinky_experiences.pkl",
        2: "models/inky_experiences.pkl",
        3: "models/clyde_experiences.pkl",
    }

    def __init__(self):
        self.experiences = {0: [], 1: [], 2: [], 3: []}
        self.models = {}
        for ghost_id, path in self.MODEL_PATHS.items():
            if os.path.exists(path):
                self.models[ghost_id] = PPO.load(path)
                print(f"{self.NOMBRES[ghost_id]} cargó su modelo — seguirá aprendiendo")
            else:
                self.models[ghost_id] = None

    def get_obs(self, ghost, player, all_ghosts) -> np.ndarray:
        """Construye la observación según el fantasma."""
        if ghost.id == 0:  # Blinky
            return np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(player.nearest_col - ghost.nearest_col),
                float(player.nearest_row - ghost.nearest_row),
            ], dtype=np.float32)

        elif ghost.id == 1:  # Pinky
            dir_col, dir_row = 0, 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            target_col = player.nearest_col + dir_col * 3
            target_row = player.nearest_row + dir_row * 3
            return np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(player.vel_x), float(player.vel_y),
                float(target_col), float(target_row),
            ], dtype=np.float32)

        elif ghost.id == 2:  # Inky
            blinky = next((g for g in all_ghosts if g.id == 0), ghost)
            dir_col, dir_row = 0, 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            pivot_col = player.nearest_col + dir_col * 2
            pivot_row = player.nearest_row + dir_row * 2
            target_col = pivot_col + (pivot_col - blinky.nearest_col)
            target_row = pivot_row + (pivot_row - blinky.nearest_row)
            return np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(blinky.nearest_col), float(blinky.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(target_col), float(target_row),
            ], dtype=np.float32)

        else:  # Clyde
            return np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
            ], dtype=np.float32)

    def get_reward(self, ghost, player, all_ghosts) -> float:
        """Calcula la recompensa según el rol del fantasma."""
        if ghost.id == 0:  # Blinky — recompensa por acercarse directo
            dist = abs(player.nearest_col - ghost.nearest_col) + \
                   abs(player.nearest_row - ghost.nearest_row)
            return 1.0 if dist <= 3 else -0.3

        elif ghost.id == 1:  # Pinky — recompensa por estar adelante
            dir_col, dir_row = 0, 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            target_col = player.nearest_col + dir_col * 3
            target_row = player.nearest_row + dir_row * 3
            dist = abs(ghost.nearest_col - target_col) + abs(ghost.nearest_row - target_row)
            return 1.0 if dist <= 2 else -0.3

        elif ghost.id == 2:  # Inky — recompensa por coordinarse con Blinky
            blinky = next((g for g in all_ghosts if g.id == 0), ghost)
            dir_col, dir_row = 0, 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            pivot_col = player.nearest_col + dir_col * 2
            pivot_row = player.nearest_row + dir_row * 2
            target_col = max(0, min(pivot_col + (pivot_col - blinky.nearest_col), 18))
            target_row = max(0, min(pivot_row + (pivot_row - blinky.nearest_row), 22))
            dist = abs(ghost.nearest_col - target_col) + abs(ghost.nearest_row - target_row)
            return 1.0 if dist <= 2 else -0.3

        else:  # Clyde — recompensa por acercarse
            dist = abs(player.nearest_col - ghost.nearest_col) + \
                   abs(player.nearest_row - ghost.nearest_row)
            return 1.0 if dist < 5 else -0.3

    def record(self, ghost, player, all_ghosts, action: int, done: bool):
        """Registra una experiencia del fantasma."""
        obs = self.get_obs(ghost, player, all_ghosts)
        reward = self.get_reward(ghost, player, all_ghosts)
        self.experiences[ghost.id].append((obs, action, reward, done))

    def save_all_experiences(self):
        """Guarda las experiencias de todos los fantasmas."""
        for ghost_id in range(4):
            exps = self.experiences[ghost_id]
            if len(exps) < 50:
                self.experiences[ghost_id] = []
                continue

            path = self.EXPERIENCE_PATHS[ghost_id]
            existing = []
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        existing = pickle.load(f)
                except Exception:
                    existing = []

            combined = (existing + exps)[-2000:]
            with open(path, 'wb') as f:
                pickle.dump(combined, f)

            nombre = self.NOMBRES[ghost_id]
            print(f"{nombre} guardó {len(exps)} experiencias nuevas ({len(combined)} totales).")
            self.experiences[ghost_id] = []

        print("Corre 'python train_ghosts_individual.py' para que todos aprendan.")