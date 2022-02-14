from typing import Tuple, Any

import gym

class RTMDP(gym.Wrapper):
    """
    A wrapper to transform any environment into a real-time environment.
    The wrapped environment is equivalent to a 1-step constant delay of the original environment.
    """

    def __init__(self, env: gym.Env, initial_action: int):
        """
        env: The environment to be wrapped into a real-time environment
        initial_action: The first action that is automatically taken in this environment
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        self.initial_action = initial_action
        self.last_action = initial_action

    def reset(self) -> Tuple[gym.spaces.Tuple, int]:
        self.last_action = self.initial_action
        s0 = super().reset()
        return s0, self.initial_action

    def step(self, action: int) -> tuple[tuple[Any, int], Any, Any, Any]:
        s, r, d, m = super().step(self.last_action)
        self.last_action = action
        return (s, action), r, d, m
