import os
import numpy as np
from pyparsing import actions
from stable_baselines3 import PPO

from src.utils import action


class TrainedGhostAgent:
    """
    Carga el modelo PPO entrenado y decide las acciones
    de los 4 fantasmas en cada paso del juego.
    """

    def __init__(self, model_path: str = "models/ghosts_ppo_final.zip"):
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"Modelo de fantasmas cargado desde {model_path}")
        else:
            self.model = None
            print("⚠️ No se encontró modelo entrenado, fantasmas usarán comportamiento base")

    def get_actions(self, ghosts, player) -> list:
        if self.model is None:
            return [None, None, None, None]

        # Asegurarse de tener exactamente 4 fantasmas
        obs = []
        for i in range(4):
            if i < len(ghosts):
                obs.append(float(ghosts[i].nearest_col))
                obs.append(float(ghosts[i].nearest_row))
            else:
                obs.append(0.0)
                obs.append(0.0)
        obs.append(float(player.nearest_col))
        obs.append(float(player.nearest_row))
        obs = np.array(obs, dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)

        actions = []
        a = int(action)
        for _ in range(4):
            actions.append(a % 4)
            a //= 4

        return actions

    def decode_to_path(self, ghost, action: int, path_finder) -> list:
        """
        Convierte una acción (0-3) en un destino y calcula el path.
        0=izquierda, 1=derecha, 2=arriba, 3=abajo
        """
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1

        if action == 0:
            target_col = max(0, ghost.nearest_col - 2)
            target_row = ghost.nearest_row
        elif action == 1:
            target_col = min(max_col, ghost.nearest_col + 2)
            target_row = ghost.nearest_row
        elif action == 2:
            target_col = ghost.nearest_col
            target_row = max(0, ghost.nearest_row - 2)
        else:
            target_col = ghost.nearest_col
            target_row = min(max_row, ghost.nearest_row + 2)

        try:
            if path_finder.state_map[target_row][target_col] == 0:
                return path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row,
                    target_col, target_row
                )
        except Exception:
            pass
        return None