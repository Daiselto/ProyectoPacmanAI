import numpy as np
import os
import pickle
from stable_baselines3 import PPO


class ClydeLearner:
    MODEL_PATH = "models/clyde_ppo.zip"
    EXPERIENCE_PATH = "models/clyde_experiences.pkl"

    def __init__(self):
        self.experiences = []
        self.model = None
        if os.path.exists(self.MODEL_PATH):
            self.model = PPO.load(self.MODEL_PATH)
            print("Clyde cargó su modelo anterior — seguirá aprendiendo")
        else:
            print("Clyde empieza desde cero")

    def get_obs(self, ghost, player) -> np.ndarray:
        return np.array([
            float(ghost.nearest_col),
            float(ghost.nearest_row),
            float(player.nearest_col),
            float(player.nearest_row),
        ], dtype=np.float32)

    def get_action(self, ghost, player) -> int:
        if self.model is None:
            return np.random.randint(0, 4)
        obs = self.get_obs(ghost, player)
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)

    def record(self, ghost, player, action: int, reward: float, done: bool):
        obs = self.get_obs(ghost, player)
        self.experiences.append((obs, action, reward, done))

    def save_experiences(self):
        """Guarda las experiencias de esta partida para entrenar después."""
        if len(self.experiences) < 50:
            self.experiences = []
            return
        # Cargar experiencias anteriores si existen
        existing = []
        if os.path.exists(self.EXPERIENCE_PATH):
            try:
                with open(self.EXPERIENCE_PATH, 'rb') as f:
                    existing = pickle.load(f)
            except Exception:
                existing = []
        # Combinar y limitar a las últimas 2000
        combined = (existing + self.experiences)[-2000:]
        with open(self.EXPERIENCE_PATH, 'wb') as f:
            pickle.dump(combined, f)
        print(f"Clyde guardó {len(self.experiences)} experiencias nuevas ({len(combined)} totales).")
        print("Corre 'python train_ghosts_individual.py' para que Clyde aprenda de ellas.")
        self.experiences = []