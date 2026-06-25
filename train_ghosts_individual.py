import os
import pygame as pg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.env.ghost_env_individual import BlinkyEnv, PinkyEnv, InkyEnv, ClydeEnv

# ── Configuración ──────────────────────────────────────────
LAYOUT = "classic"
N_EPISODES = 30
MODEL_DIR = "models"
LOG_DIR = "logs"
# ───────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, nombre):
        super().__init__()
        self.nombre = nombre
        self.episode = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode += 1
            print(f"  [{self.nombre}] Episodio {self.episode:3d}/{N_EPISODES} | "
                  f"Recompensa: {self.episode_reward:8.2f}")
            self.episode_reward = 0.0
        return True


def entrenar_fantasma(nombre, EnvClass, obs_shape, model_file):
    print(f"\n{'='*50}")
    print(f"  Entrenando {nombre}...")
    print(f"{'='*50}")

    env = EnvClass(layout=LAYOUT)

    if os.path.exists(model_file):
        print(f"  Cargando modelo existente: {model_file}")
        model = PPO.load(model_file, env=env)
    else:
        print(f"  Creando nuevo modelo PPO para {nombre}...")
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

    total_steps = N_EPISODES * 1000
    model.learn(
        total_timesteps=total_steps,
        callback=EpisodeLoggerCallback(nombre),
        reset_num_timesteps=False
    )

    model.save(model_file)
    print(f"\n  ✅ {nombre} guardado en {model_file}")
    env.close()


if __name__ == "__main__":
    pg.display.init()
    pg.display.set_mode((1, 1), pg.NOFRAME)

    print("\n" + "="*50)
    print("  Entrenamiento Individual de Fantasmas con PPO")
    print(f"  Episodios por fantasma: {N_EPISODES}")
    print("="*50)

    entrenar_fantasma("Blinky", BlinkyEnv, 6,  os.path.join(MODEL_DIR, "blinky_ppo.zip"))
    entrenar_fantasma("Pinky",  PinkyEnv,  8,  os.path.join(MODEL_DIR, "pinky_ppo.zip"))
    entrenar_fantasma("Inky",   InkyEnv,   8,  os.path.join(MODEL_DIR, "inky_ppo.zip"))
    entrenar_fantasma("Clyde",  ClydeEnv,  4,  os.path.join(MODEL_DIR, "clyde_ppo.zip"))

    print("\n✅ Entrenamiento completo. Todos los modelos guardados en /models")