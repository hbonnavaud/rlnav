from gym.spaces import Box
from gymnasium import Env
import numpy as np
import importlib
import random
from enum import Enum
from matplotlib import pyplot as plt
from typing import Any


class Colors(Enum):
    EMPTY = [250, 250, 250]
    WALL = [50, 54, 51]
    START = [213, 219, 214]
    TRAP = [73, 179, 101]
    TILE_BORDER = [50, 54, 51]
    AGENT = [0, 0, 255]
    GOAL = [255, 0, 0]


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    REWARD = 3
    TRAP = 4


class PointMazeV0(Env):
    """
    2D Point Maze Navigation Environment.
    In the code bellow, x/y refers to the agent's position in the observation space, i/j refers to the agent's position
    in the maze grid (i refers to the row id, j to the column id).
    """

    name = "Point-Maze"

    def __init__(self, **params):
        self.map_name = params.get("map_name", "EMPTY")
        self.action_noise = params.get("action_noise", 1.0)
        self.reset_anywhere = params.get("reset_anywhere", True)
        self.goal_conditioned = params.get("goal_conditioned", False)

        assert isinstance(self.action_noise, float) and self.action_noise > 0, "Invalid action_noise value."
        assert isinstance(self.reset_anywhere, bool), "reset_anywhere must be a boolean."
        assert isinstance(self.goal_conditioned, bool), "goal_conditioned must be a boolean."

        """Loads the maze map from the given map name."""
        module_path = f"sciborg.environments.point_maze.maps.{self.map_name}"
        self.maze_array = np.array(importlib.import_module(module_path).maze_array, dtype=np.float16)
        self.height, self.width = self.maze_array.shape

        """Defines the observation and action spaces."""
        self.observation_space = Box(
            low=np.array([-self.width / 2, -self.height / 2], dtype=np.float32),
            high=np.array([self.width / 2, self.height / 2], dtype=np.float32)
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        self.agent_observation = None
        if self.goal_conditioned:
            self.goal = None
        self.reset()

    def _sample_reachable_position(self) -> np.ndarray:
        empty_tiles = np.argwhere(
            np.logical_or(self.maze_array == TileType.EMPTY.value,
                          self.maze_array == TileType.REWARD.value))
        if len(empty_tiles) == 0:
            raise ValueError("Looking for a reachable position but none was found.")

        # Sample a reachable tile
        reachable_tile = random.choice(empty_tiles)
        # Reachable position at the center of the tile
        position = self.get_observation(*reachable_tile)
        # Sample a point in the selected tile, avoid np.random to keep the reset seed for random package
        noise = np.array([random.random(), random.random()]) - 0.5
        return position + noise

    def get_observation(self, i: int, j: int) -> np.ndarray:
        """Converts grid coordinates to an observation that belongs to the center of the tile."""
        return np.array([j + 0.5 - self.width / 2, -(i + 0.5 - self.height / 2)])

    def get_coordinates(self, observation: np.ndarray) -> tuple[int, int]:
        """Return the tile that belongs to the given observation."""
        return int(- observation[1] + self.height / 2), int(observation[0] + self.width / 2)

    def reset(self, *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        """Resets the agent's position."""
        if self.reset_anywhere:
            self.agent_observation = self._sample_reachable_position()
        else:
            valid_tiles = np.argwhere(self.maze_array == 2)
            if len(valid_tiles) == 0:
                raise ValueError("Cannot reset with reset_anywhere=False and no available start tiles.")
            self.agent_observation = self.get_observation(*random.choice(valid_tiles))
        assert self.observation_space.contains(self.agent_observation.astype(self.observation_space.dtype))

        """Sample a goal."""
        if self.goal_conditioned:
            self.goal = self._sample_reachable_position()
            return self.agent_observation.copy(), self.goal.copy()

        return self.agent_observation.copy()

    def is_available(self, i: int, j: int) -> bool:
        """Checks if a position (i, j) in the maze array is available (not a wall or out of bounds)."""
        return 0 <= j < self.width and 0 <= i < self.height and self.maze_array[i, j] != TileType.WALL.value

    def step(self, action: np.ndarray):
        """Moves the agent and returns the new state."""
        assert self.action_space.contains(action.astype(self.action_space.dtype))

        action = np.clip(action + np.random.normal(0, self.action_noise),
                         self.action_space.low, self.action_space.high)

        for _ in range(10):  # Sub-steps for smooth movement
            new_observation = self.agent_observation + action / 10
            if self.is_available(*self.get_coordinates(new_observation)):
                self.agent_observation = new_observation

        tile_type = self.maze_array[tuple(self.get_coordinates(self.agent_observation))]
        reward = 10 if tile_type == TileType.REWARD.value else -1
        return self.agent_observation.copy(), reward, False, {}

    def render(self):
        """Renders the maze environment."""
        img = np.zeros((self.height * 10, self.width * 10, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                img[y * 10:(y + 1) * 10, x * 10:(x + 1) * 10] = Colors.WALL.value if self.maze_map[
                                                                                         y, x] == 1 else Colors.EMPTY.value
        return img

    def plot(self):
        """Plots the environment in a matplotlib window."""
        plt.imshow(self.render())
        plt.axis('off')
        plt.show()

    def copy(self):
        return PointEnv(map_name=self.map_name, action_noise=self.action_noise, reset_anywhere=self.reset_anywhere,
                        goal_conditioned=self.goal_conditioned)
