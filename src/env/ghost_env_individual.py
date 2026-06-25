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


class BlinkyEnv(gym.Env):
    """Blinky aprende a perseguir directamente a Pac-Man."""

    def __init__(self, layout="classic"):
        super().__init__()
        self.layout = layout
        pg.display.init()
        pg.display.set_mode((1, 1), pg.NOFRAME)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(maze=self.maze, screen=None, sounds_active=False, state_active=False, agent=None)
        self.ghost_id = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(6,), dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0
        self.prev_dist = 0

    def _get_obs(self):
        ghost = self.game.ghosts[self.ghost_id]
        player = self.game.player
        return np.array([
            float(ghost.nearest_col), float(ghost.nearest_row),
            float(player.nearest_col), float(player.nearest_row),
            float(player.nearest_col - ghost.nearest_col),
            float(player.nearest_row - ghost.nearest_row),
        ], dtype=np.float32)

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
        ghost = self.game.ghosts[self.ghost_id]
        player = self.game.player
        self.prev_dist = abs(player.nearest_col - ghost.nearest_col) + abs(player.nearest_row - ghost.nearest_row)
        return self._get_obs(), {}

    def _apply_action(self, action):
        ghost = self.game.ghosts[self.ghost_id]
        path_finder = self.game.path_finder
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1
        moves = {
            0: (max(0, ghost.nearest_col - 2), ghost.nearest_row),
            1: (min(max_col, ghost.nearest_col + 2), ghost.nearest_row),
            2: (ghost.nearest_col, max(0, ghost.nearest_row - 2)),
            3: (ghost.nearest_col, min(max_row, ghost.nearest_row + 2)),
        }
        target_col, target_row = moves[action]
        try:
            if path_finder.state_map[target_row][target_col] == 0:
                ghost.current_path = path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row, target_col, target_row)
        except Exception:
            pass

    def step(self, action):
        self.current_step += 1
        self.game.player.change_player_vel(Action(random.randint(0, 3)), self.game)
        self._apply_action(int(action))
        self.game.move_players()

        ghost = self.game.ghosts[self.ghost_id]
        player = self.game.player
        dist = abs(player.nearest_col - ghost.nearest_col) + abs(player.nearest_row - ghost.nearest_row)

        # Blinky: recompensa por acercarse directo
        reward = 1.0 if dist < self.prev_dist else -0.5
        if dist <= 1:
            reward = 15.0

        self.prev_dist = dist
        done = (self.game.game_mode in [GameMode.game_over, GameMode.hit_ghost] or
                dist <= 1 or self.current_step >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def close(self):
        pass


class PinkyEnv(gym.Env):
    """Pinky aprende a posicionarse adelante de Pac-Man."""

    def __init__(self, layout="classic"):
        super().__init__()
        self.layout = layout
        pg.display.init()
        pg.display.set_mode((1, 1), pg.NOFRAME)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(maze=self.maze, screen=None, sounds_active=False, state_active=False, agent=None)
        self.ghost_id = 1
        self.action_space = spaces.Discrete(4)
        # obs: ghost pos, player pos, player vel, target pos (adelante del player)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0

    def _get_target(self):
        """3 casillas adelante de Pac-Man según su velocidad."""
        player = self.game.player
        dir_col = 0
        dir_row = 0
        if player.vel_x > 0:
            dir_col = 1
        elif player.vel_x < 0:
            dir_col = -1
        elif player.vel_y > 0:
            dir_row = 1
        elif player.vel_y < 0:
            dir_row = -1
        return player.nearest_col + dir_col * 3, player.nearest_row + dir_row * 3

    def _get_obs(self):
        ghost = self.game.ghosts[self.ghost_id]
        player = self.game.player
        target_col, target_row = self._get_target()
        return np.array([
            float(ghost.nearest_col), float(ghost.nearest_row),
            float(player.nearest_col), float(player.nearest_row),
            float(player.vel_x), float(player.vel_y),
            float(target_col), float(target_row),
        ], dtype=np.float32)

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
        return self._get_obs(), {}

    def _apply_action(self, action):
        ghost = self.game.ghosts[self.ghost_id]
        path_finder = self.game.path_finder
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1
        moves = {
            0: (max(0, ghost.nearest_col - 2), ghost.nearest_row),
            1: (min(max_col, ghost.nearest_col + 2), ghost.nearest_row),
            2: (ghost.nearest_col, max(0, ghost.nearest_row - 2)),
            3: (ghost.nearest_col, min(max_row, ghost.nearest_row + 2)),
        }
        target_col, target_row = moves[action]
        try:
            if path_finder.state_map[target_row][target_col] == 0:
                ghost.current_path = path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row, target_col, target_row)
        except Exception:
            pass

    def step(self, action):
        self.current_step += 1
        self.game.player.change_player_vel(Action(random.randint(0, 3)), self.game)
        self._apply_action(int(action))
        self.game.move_players()

        ghost = self.game.ghosts[self.ghost_id]
        target_col, target_row = self._get_target()

        # Pinky: recompensa por estar cerca del punto adelante de Pac-Man
        dist_to_target = abs(ghost.nearest_col - target_col) + abs(ghost.nearest_row - target_row)
        reward = 1.0 if dist_to_target <= 2 else -0.3
        if dist_to_target == 0:
            reward = 10.0

        done = (self.game.game_mode in [GameMode.game_over, GameMode.hit_ghost] or
                self.current_step >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def close(self):
        pass


class InkyEnv(gym.Env):
    """Inky aprende a coordinarse con Blinky para acorralar a Pac-Man."""

    def __init__(self, layout="classic"):
        super().__init__()
        self.layout = layout
        pg.display.init()
        pg.display.set_mode((1, 1), pg.NOFRAME)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(maze=self.maze, screen=None, sounds_active=False, state_active=False, agent=None)
        self.ghost_id = 2
        self.action_space = spaces.Discrete(4)
        # obs: inky pos, blinky pos, player pos, target calculado
        self.observation_space = spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0
    def _get_target(self):
        """Calcula el objetivo de Inky basado en Blinky y Pac-Man."""
        player = self.game.player
        blinky = self.game.ghosts[0]
        dir_col = 0
        dir_row = 0
        if player.vel_x > 0:
            dir_col = 1
        elif player.vel_x < 0:
            dir_col = -1
        elif player.vel_y > 0:
            dir_row = 1
        elif player.vel_y < 0:
            dir_row = -1
        pivot_col = player.nearest_col + dir_col * 2
        pivot_row = player.nearest_row + dir_row * 2
        target_col = pivot_col + (pivot_col - blinky.nearest_col)
        target_row = pivot_row + (pivot_row - blinky.nearest_row)
        max_row = self.game.path_finder.state_map.shape[0] - 1
        max_col = self.game.path_finder.state_map.shape[1] - 1
        return max(0, min(target_col, max_col)), max(0, min(target_row, max_row))

    def _get_obs(self):
        ghost = self.game.ghosts[self.ghost_id]
        blinky = self.game.ghosts[0]
        player = self.game.player
        target_col, target_row = self._get_target()
        return np.array([
            float(ghost.nearest_col), float(ghost.nearest_row),
            float(blinky.nearest_col), float(blinky.nearest_row),
            float(player.nearest_col), float(player.nearest_row),
            float(target_col), float(target_row),
        ], dtype=np.float32)

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
        return self._get_obs(), {}

    def _apply_action(self, action):
        ghost = self.game.ghosts[self.ghost_id]
        path_finder = self.game.path_finder
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1
        moves = {
            0: (max(0, ghost.nearest_col - 2), ghost.nearest_row),
            1: (min(max_col, ghost.nearest_col + 2), ghost.nearest_row),
            2: (ghost.nearest_col, max(0, ghost.nearest_row - 2)),
            3: (ghost.nearest_col, min(max_row, ghost.nearest_row + 2)),
        }
        target_col, target_row = moves[action]
        try:
            if path_finder.state_map[target_row][target_col] == 0:
                ghost.current_path = path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row, target_col, target_row)
        except Exception:
            pass

    def step(self, action):
        self.current_step += 1
        self.game.player.change_player_vel(Action(random.randint(0, 3)), self.game)
        self._apply_action(int(action))
        self.game.move_players()

        ghost = self.game.ghosts[self.ghost_id]
        target_col, target_row = self._get_target()

        # Inky: recompensa por llegar al punto de coordinación con Blinky
        dist_to_target = abs(ghost.nearest_col - target_col) + abs(ghost.nearest_row - target_row)
        reward = 1.0 if dist_to_target <= 2 else -0.3
        if dist_to_target == 0:
            reward = 10.0

        done = (self.game.game_mode in [GameMode.game_over, GameMode.hit_ghost] or
                self.current_step >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def close(self):
        pass


class ClydeEnv(gym.Env):
    """Clyde aprende a moverse pero con recompensas bajas — sigue siendo impredecible."""

    def __init__(self, layout="classic"):
        super().__init__()
        self.layout = layout
        pg.display.init()
        pg.display.set_mode((1, 1), pg.NOFRAME)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(maze=self.maze, screen=None, sounds_active=False, state_active=False, agent=None)
        self.ghost_id = 3
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(4,), dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0

    def _get_obs(self):
        ghost = self.game.ghosts[self.ghost_id]
        player = self.game.player
        return np.array([
            float(ghost.nearest_col), float(ghost.nearest_row),
            float(player.nearest_col), float(player.nearest_row),
        ], dtype=np.float32)

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
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.game.player.change_player_vel(Action(random.randint(0, 3)), self.game)

        ghost = self.game.ghosts[self.ghost_id]
        path_finder = self.game.path_finder
        max_row = path_finder.state_map.shape[0] - 1
        max_col = path_finder.state_map.shape[1] - 1
        moves = {
            0: (max(0, ghost.nearest_col - 2), ghost.nearest_row),
            1: (min(max_col, ghost.nearest_col + 2), ghost.nearest_row),
            2: (ghost.nearest_col, max(0, ghost.nearest_row - 2)),
            3: (ghost.nearest_col, min(max_row, ghost.nearest_row + 2)),
        }
        target_col, target_row = moves[int(action)]
        try:
            if path_finder.state_map[target_row][target_col] == 0:
                ghost.current_path = path_finder.get_min_path(
                    ghost.nearest_col, ghost.nearest_row, target_col, target_row)
        except Exception:
            pass
        self.game.move_players()

        # Clyde: recompensa muy baja y aleatoria — sigue siendo impredecible
        reward = random.uniform(-0.5, 0.5)

        done = (self.game.game_mode in [GameMode.game_over, GameMode.hit_ghost] or
                self.current_step >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def close(self):
        pass