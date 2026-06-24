from typing import Union, Tuple, Any, Dict

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces

from src.controller import Controller
from src.game import Game
from src.map import Map
from src.utils.action import Action
from src.utils.game_mode import GameMode


class PacmanEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    reward_range = (-10, 5)

    def __init__(self, layout: str, enable_render=True, state_active=False, player_lives: int = 3):
        self.layout = layout
        self.state_active = state_active
        self.enable_render = enable_render
        if enable_render:
            pg.init()

        self.action_space = spaces.Discrete(len(Action))
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(
            maze=self.maze,
            screen=Controller.get_screen(state_active, self.width, self.height) if enable_render else None,
            sounds_active=False,
            state_active=state_active,
            agent=None
        )
        self.timer = 0
        self.reinit_game = False
        self.player_lives = player_lives

        screen_array = self.get_screen_rgb_array()
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=screen_array.shape,
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.maze.reinit_map()
        self.game.restart()
        self.game.player.regenerate()
        self.game.score = 0
        self.game.mode_timer = 0
        self.game.ghosts_timer = 0
        self.game.set_mode(GameMode.normal)
        self.game.make_ghosts_normal()
        self.game.player.lives = self.player_lives
        obs = self.get_screen_rgb_array()
        info = self.get_info_dict()
        return obs, info

    def render(self):
        if self.enable_render:
            self.game.init_screen()
            self.game.draw()
            pg.display.flip()

    def close(self):
        if self.enable_render:
            pg.quit()

    def step(self, action: Union[Action, int]):
        action = Action(int(action)) if type(action) is int else action
        reward = self._one_step_action(action)
        done = self.get_mode() in [GameMode.game_over, GameMode.black_screen]
        obs = self.get_screen_rgb_array()
        info = self.get_info_dict()
        return obs, reward, done, False, info

    def get_info_dict(self) -> Dict[str, Any]:
        ghosts_pixel_pos = [ghost.get_pixel_position() for ghost in self.game.ghosts]
        number_of_scared_ghosts = sum([ghost.is_vulnerable() for ghost in self.game.ghosts])
        return {
            'win': self.get_mode() == GameMode.black_screen,
            'player_position': self.get_player_position(),
            'player_pixel_position': self.get_player_pixel_position(),
            'player_lives': self.game.player.lives,
            'game_mode': self.get_mode().value,
            'game_score': self.game.score,
            'scared_ghosts': number_of_scared_ghosts,
            'ghosts_pixel_pos': ghosts_pixel_pos,
            'player_vel': self.game.player.get_vel(),
            'player_action': self.game.player.current_action
        }

    def _one_step_action(self, action: Union[Action, int]) -> int:
        self.check_game_mode()

        if self.get_mode() in [GameMode.game_over, GameMode.black_screen]:
            return 0
        if self.reinit_game:
            self.reinit_game = False
            return 0

        prev_reward = self.game.total_rewards
        self.game.player.change_player_vel(action, self.game)
        self.game.move_players()
        succ_reward = self.game.total_rewards
        return succ_reward - prev_reward

    def get_mode(self) -> GameMode:
        return self.game.game_mode

    def get_state_matrix(self) -> np.ndarray:
        return self.maze.state_matrix

    def get_player_position(self) -> Tuple[int, int]:
        return self.game.player.nearest_col, self.game.player.nearest_row

    def get_player_pixel_position(self) -> Tuple[int, int]:
        return self.game.player.x, self.game.player.y

    def check_game_mode(self):
        mode = self.get_mode()

        if self.maze.get_number_of_pellets() == 0:
            self.game.set_mode(GameMode.black_screen)
            return

        if mode is GameMode.hit_ghost:
            self.game.player.lives -= 1
            if self.game.player.lives == 0:
                self.game.set_mode(GameMode.game_over)
            else:
                self.game.init_players_in_map()
                self.game.make_ghosts_normal()
                self.game.set_mode(GameMode.normal)
                self.reinit_game = True
        elif mode == GameMode.wait_after_eating_ghost:
            self.game.move_ghosts()
            if self.maze.get_number_of_pellets() == 0:
                self.game.set_mode(GameMode.black_screen)
            elif self.game.are_all_ghosts_vulnerable():
                self.game.set_mode(GameMode.change_ghosts)
            elif self.game.are_all_ghosts_normal():
                self.game.set_mode(GameMode.normal)

        self.game.check_ghosts_state()

    def get_screen_rgb_array(self):
        if self.game.screen is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        screen = self.game.screen.copy()
        return pg.surfarray.pixels3d(screen)