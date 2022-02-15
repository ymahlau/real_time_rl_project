from typing import Tuple

import gym
import numpy as np

"""
Environments are from the following article: https://andyljones.com/posts/rl-debugging.html#probe
"""

class ConstRewardEnv(gym.Env):
    """
    Environment with one action, one observation, one timestep and one reward (+1).
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([0]), dtype=int)

    def reset(self):
        return [0]

    def step(self, action: int):
        return [0], 1, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()


class PredictableRewardEnv(gym.Env):
    """
    Environment with one action, two observation (+1/-1), one time step and observation-dependent reward (+1/-1).
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=int)

        self.num = np.random.choice([-1, 1])

    def reset(self):
        self.num = np.random.choice([-1, 1])
        return [self.num]

    def step(self, action: int) -> Tuple[list, int, bool, dict]:
        return [self.num], self.num, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()


class TwoStepsEnv(gym.Env):
    """
    Environment with one action, two observations (0, then 1), two timesteps and reward (+1) at the end.
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=int)

        self.time = 0

    def reset(self):
        self.time = 0
        return [0]

    def step(self, action: int) -> Tuple[list, int, bool, dict]:
        if self.time == 0:
            self.time += 1
            return [1], 0, False, {}
        else:
            return [0], 1, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()


class TwoActionsEnv(gym.Env):
    """
    Environment with two actions (0/1), observation, one timestep and action-dependent reward (+1/-1).
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([0]), dtype=int)

    def reset(self):
        return [0]

    def step(self, action: int) -> Tuple[list, int, bool, dict]:
        if action == 0:
            return [0], -1, True, {}
        else:
            return [0], 1, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()


class TwoStatesActionsEnv(gym.Env):
    """
    Environment with two actions (0/1), two observation (-1/+1), one timestep and (action & observation)-dependent
    reward (+1/-1).
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=int)

        self.state = np.random.choice([-1, 1])

    def reset(self):
        self.state = np.random.choice([-1, 1])
        return [self.state]

    def step(self, action: int) -> Tuple[list, int, bool, dict]:
        if action == 0 and self.state == -1:
            return [0], -1, True, {}
        elif action == 1 and self.state == -1:
            return [1], 1, True, {}
        elif action == 0 and self.state == 1:
            return [1], 1, True, {}
        elif action == 1 and self.state == 1:
            return [0], -1, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()


class TwoActionsTwoStates(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([4]), dtype=int)
        self.state = np.random.choice([0, 1])

    def reset(self):
        self.state = np.random.choice([0, 1])
        return [self.state]

    def step(self, action: int) -> Tuple[list, int, bool, dict]:
        if self.state == 0:
            self.state = 2
            return [self.state], 1, False, {}
        elif self.state == 1:
            self.state = 3
            return [self.state], 1, False, {}
        elif self.state == 2:
            self.state = 4
            return [self.state], 1 if action == 1 else 0, True, {}
        elif self.state == 3:
            self.state = 4
            return [self.state], 0 if action == 1 else 1, True, {}

    def close(self):
        return True

    def render(self, mode: str = "human"):
        raise NotImplementedError()
