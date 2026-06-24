import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame as pg
import random

from src.controller import Controller
from src.game import Game
from src.map import Map
from src.utils.game_mode import GameMode
from src.utils.action import Action
from src.utils.ghost_state import GhostState


class GhostEnv(gym.Env):
    """
    Entorno Gymnasium donde el agente controla los 4 fantasmas simultáneamente.
    Pac-Man se mueve aleatoriamente durante el entrenamiento.
    Objetivo: los fantasmas aprenden a coordinarse para atrapar a Pac-Man.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, layout: str = "classic", enable_render: bool = False):
        super().__init__()
        self.layout = layout
        self.enable_render = enable_render

        pg.display.init()
        if enable_render:
            pg.init()
            self.maze = Map(layout)
            self.width, self.height = self.maze.get_map_sizes()
            self.screen = Controller.get_screen(False, self.width, self.height)
        else:
            pg.display.set_mode((1, 1), pg.NOFRAME)
            self.maze = Map(layout)
            self.width, self.height = self.maze.get_map_sizes()
            self.screen = None

        self.game = Game(
            maze=self.maze,
            screen=self.screen,
            sounds_active=False,
            state_active=False,
            agent=None
        )

        # 4 fantasmas x 4 direcciones = 256 combinaciones posibles
        # Cada acción se decodifica en 4 movimientos individuales
        self.action_space = spaces.Discrete(256)

        # Observación: para cada fantasma [col, row] + posición de Pac-Man [col, row]
        # = 4*2 + 2 = 10 valores
        self.observation_space = spaces.Box(
            low=-50, high=50,
            shape=(10,),
            dtype=np.float32
        )

        self.max_steps = 1000
        self.current_step = 0
        self.prev_dists = [0, 0, 0, 0]

    def _decode_action(self, action: int):
        """
        Decodifica un número 0-255 en 4 acciones individuales (0-3 cada una).
        Usa base 4: acción = a0 + a1*4 + a2*16 + a3*64
        """
        actions = []
        for _ in range(4):
            actions.append(action % 4)
            action //= 4
        return actions  # [accion_blinky, accion_pinky, accion_inky, accion_clyde]

    def _get_obs(self):
        player = self.game.player
        obs = []
        for ghost in self.game.ghosts:
            obs.append(float(ghost.nearest_col))
            obs.append(float(ghost.nearest_row))
        obs.append(float(player.nearest_col))
        obs.append(float(player.nearest_row))
        return np.array(obs, dtype=np.float32)

    def _move_ghost_toward_target(self, ghost, target_col, target_row):
        path_finder = self.game.path_finder
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1
        target_col = max(0, min(target_col, max_col))
        target_row = max(0, min(target_row, max_row))
        try:
            if path_finder.state_map[target_row][target_col] == 0:
                ghost.current_path = path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row,
                    target_col, target_row
                )
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze.reinit_map()
        self.game.restart()
        self.game.player.regenerate()
        self.game.score = 0
        self.game.mode_timer = 0
        self.game.ghosts_timer = 0
        self.game.set_mode(GameMode.normal)
        self.game.make_ghosts_normal()
        self.game.player.lives = 3
        self.current_step = 0

        player = self.game.player
        self.prev_dists = [
            abs(player.nearest_col - g.nearest_col) + abs(player.nearest_row - g.nearest_row)
            for g in self.game.ghosts
        ]

        return self._get_obs(), {}

    def step(self, action: int):
        self.current_step += 1

        # Pac-Man se mueve aleatoriamente
        random_action = Action(random.randint(0, 3))
        self.game.player.change_player_vel(random_action, self.game)

        # Decodificar acción en 4 movimientos individuales
        ghost_actions = self._decode_action(int(action))

        # Aplicar acción a cada fantasma
        for i, ghost in enumerate(self.game.ghosts):
            a = ghost_actions[i]
            if a == 0:   # izquierda
                target_col = ghost.nearest_col - 2
                target_row = ghost.nearest_row
            elif a == 1: # derecha
                target_col = ghost.nearest_col + 2
                target_row = ghost.nearest_row
            elif a == 2: # arriba
                target_col = ghost.nearest_col
                target_row = ghost.nearest_row - 2
            else:        # abajo
                target_col = ghost.nearest_col
                target_row = ghost.nearest_row + 2

            self.game.ghosts[i].current_path = None
            self._move_ghost_toward_target(ghost, target_col, target_row)

        # Mover jugadores
        self.game.move_players()

        # Calcular recompensa
        player = self.game.player
        reward = 0.0
        caught = False

        for i, ghost in enumerate(self.game.ghosts):
            dist = abs(player.nearest_col - ghost.nearest_col) + \
                   abs(player.nearest_row - ghost.nearest_row)

            if dist <= 1:
                reward += 10.0  # atrapó a Pac-Man
                caught = True
            elif dist < self.prev_dists[i]:
                reward += 1.0   # se acercó
            elif dist > self.prev_dists[i]:
                reward -= 0.5   # se alejó

            self.prev_dists[i] = dist

        # Bonus por coordinación: si varios fantasmas están cerca a la vez
        cerca = sum(1 for g in self.game.ghosts
                    if abs(player.nearest_col - g.nearest_col) +
                       abs(player.nearest_row - g.nearest_row) <= 4)
        if cerca >= 2:
            reward += cerca * 0.5

        done = (
            self.game.game_mode in [GameMode.game_over, GameMode.hit_ghost] or
            caught or
            self.current_step >= self.max_steps
        )

        if self.enable_render:
            self.game.init_screen()
            self.game.draw()
            pg.display.flip()

        return self._get_obs(), reward, done, False, {}

    def close(self):
        if self.enable_render:
            pg.quit()