import os
import pygame as pg
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env.pacman_env import PacmanEnv

# ── Configuración ──────────────────────────────────────────
LAYOUT = "classic"
TOTAL_STEPS = 500_000      # pasos de entrenamiento
SAVE_EVERY = 50_000        # guardar checkpoint cada N pasos
MODEL_DIR = "models"
LOG_DIR = "logs"
# ───────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

pg.display.init()
pg.display.set_mode((1, 1), pg.NOFRAME)


def make_env():
    return PacmanEnv(layout=LAYOUT, enable_render=False)


if __name__ == "__main__":
    print("Creando entorno de entrenamiento...")
    env = make_vec_env(make_env, n_envs=1)

    print("Iniciando entrenamiento con PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_EVERY,
        save_path=MODEL_DIR,
        name_prefix="pacman_ppo"
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(os.path.join(MODEL_DIR, "pacman_ppo_final"))
    print(f"\nModelo guardado en {MODEL_DIR}/pacman_ppo_final.zip")
    env.close()