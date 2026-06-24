import os
import pygame as pg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.env.ghost_env import GhostEnv

# ── Configuración ──────────────────────────────────────────
LAYOUT = "classic"
N_EPISODES = 30
MODEL_DIR = "models"
LOG_DIR = "logs"
# ───────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class EpisodeLoggerCallback(BaseCallback):
    """
    Muestra información de cada episodio durante el entrenamiento.
    """
    def __init__(self):
        super().__init__()
        self.episode = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode += 1
            print(f"Episodio {self.episode:3d}/{N_EPISODES} | "
                  f"Recompensa: {self.episode_reward:8.2f} | "
                  f"Pasos: {self.num_timesteps}")
            self.episode_reward = 0.0
        return True


if __name__ == "__main__":
    pg.display.init()
    pg.display.set_mode((1, 1), pg.NOFRAME)

    print("=" * 50)
    print("  Entrenamiento de Fantasmas con PPO")
    print(f"  Episodios: {N_EPISODES}")
    print("=" * 50)

    env = GhostEnv(layout=LAYOUT, enable_render=False)

    # Cargar modelo existente si ya fue entrenado antes
    model_path = os.path.join(MODEL_DIR, "ghosts_ppo_final.zip")
    if os.path.exists(model_path):
        print(f"\nCargando modelo existente desde {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("\nCreando nuevo modelo PPO...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
        )

    # Calcular pasos totales basado en episodios
    total_steps = N_EPISODES * 1000

    print(f"\nIniciando entrenamiento ({total_steps} pasos totales)...\n")
    model.learn(
        total_timesteps=total_steps,
        callback=EpisodeLoggerCallback(),
        reset_num_timesteps=False
    )

    model.save(model_path)
    print(f"\n✅ Modelo guardado en {model_path}")
    print("Ya puedes correr el juego normal y los fantasmas usarán lo aprendido.")
    env.close()