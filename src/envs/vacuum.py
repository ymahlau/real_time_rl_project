from typing import Tuple

import gym
import numpy as np


class VacuumEnv(gym.Env):
    """
        Collect poop and bring it back to the original position.

        InitialState: (for N = 4) You=0, Poop=1
        0###
        ####
        ####
        ###1

        Poop collected:
        ####
        ####
        ####
        ###0

        Poop retrieved:
        0###
        ####
        ####
        ####
    """

    def __init__(self, N: int):
        self.poopCleaned = None
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([N, N])
        self.reward_range = (0, 10)
        self.state = [0, 0]
        self.N = N  # magic number

    def reset(self):
        state = [0, 0]
        self.poopCleaned = False
        return np.array(state)

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        if action not in {0, 1, 2, 3}:
            raise Exception("invalid action")
        # This should move the vacuum
        x, y = self.state
        if action == 0:
            x -= 1
        elif action == 1:
            y -= 1
        elif action == 2:
            x += 1
        elif action == 3:
            y += 1
        x = min(max(0, x), self.N - 1)
        y = min(max(0, y), self.N - 1)
        self.state = [x, y]

        reward = -1
        if (x, y) == (self.N - 1, self.N - 1) and not self.poopCleaned:
            reward += 10
            self.poopCleaned = True

        done = False
        if self.poopCleaned and (x, y) == (0, 0):
            reward += 10
            done = True
        meta_info = {}
        return np.array(self.state), reward, done, meta_info

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()
