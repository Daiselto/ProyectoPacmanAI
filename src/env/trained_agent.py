import os
import numpy as np
from stable_baselines3 import PPO


class TrainedGhostAgent:
    """
    Carga los 4 modelos PPO individuales y decide las acciones
    de cada fantasma por separado.
    """

    def __init__(self):
        self.models = {}
        nombres = {0: "blinky", 1: "pinky", 2: "inky", 3: "clyde"}
        for ghost_id, nombre in nombres.items():
            path = f"models/{nombre}_ppo.zip"
            if os.path.exists(path):
                self.models[ghost_id] = PPO.load(path)
                print(f"Modelo {nombre} cargado desde {path}")
            else:
                self.models[ghost_id] = None
                print(f"⚠️ No se encontró modelo para {nombre}")

    def get_action(self, ghost, player, all_ghosts) -> int:
        """Devuelve la acción para un fantasma específico."""
        model = self.models.get(ghost.id)
        if model is None:
            return None

        if ghost.id == 0:  # Blinky
            obs = np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(player.nearest_col - ghost.nearest_col),
                float(player.nearest_row - ghost.nearest_row),
            ], dtype=np.float32)

        elif ghost.id == 1:  # Pinky
            dir_col = 0
            dir_row = 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            target_col = player.nearest_col + dir_col * 3
            target_row = player.nearest_row + dir_row * 3
            obs = np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(player.vel_x), float(player.vel_y),
                float(target_col), float(target_row),
            ], dtype=np.float32)

        elif ghost.id == 2:  # Inky
            blinky = next((g for g in all_ghosts if g.id == 0), ghost)
            dir_col = 0
            dir_row = 0
            if player.vel_x > 0: dir_col = 1
            elif player.vel_x < 0: dir_col = -1
            elif player.vel_y > 0: dir_row = 1
            elif player.vel_y < 0: dir_row = -1
            pivot_col = player.nearest_col + dir_col * 2
            pivot_row = player.nearest_row + dir_row * 2
            target_col = pivot_col + (pivot_col - blinky.nearest_col)
            target_row = pivot_row + (pivot_row - blinky.nearest_row)
            obs = np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(blinky.nearest_col), float(blinky.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
                float(target_col), float(target_row),
            ], dtype=np.float32)

        else:  # Clyde
            obs = np.array([
                float(ghost.nearest_col), float(ghost.nearest_row),
                float(player.nearest_col), float(player.nearest_row),
            ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=False)
        return int(action)

    def decode_to_path(self, ghost, action: int, path_finder) -> list:
        """Convierte una acción (0-3) en un path."""
        if action is None:
            return None
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