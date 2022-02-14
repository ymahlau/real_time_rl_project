import random
from collections import deque
from typing import Union, Any

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.replay_buffer = deque(maxlen=capacity)

    def add_data(self, data: Any):
        self.replay_buffer.append(data)

    def capacity_reached(self):
        return len(self.replay_buffer) >= self.replay_buffer.maxlen

    def sample(self, sample_size: int) -> Any:
        return random.sample(self.replay_buffer, sample_size)


@torch.no_grad()
def moving_average(target_params, current_params, factor):
    for t, c in zip(target_params, current_params):
        t += factor * (c - t)


def flatten_rtmdp_obs(obs: Union[np.ndarray, Tensor], num_actions: int) -> list[Any]:
    """
    Converts the observation tuple (s,a) returned by rtmdp
    into a single sequence s + one_hot_encoding(a)
    """
    # one-hot
    one_hot = np.zeros(num_actions)
    one_hot[obs[1]] = 1
    return list(obs[0]) + list(one_hot)


def evaluate_policy(policy, env, trials=10, rtmdp_ob=True) -> float:
    cum_rew = 0
    for _ in range(trials):
        state = env.reset()
        done = False
        while not done:
            if rtmdp_ob:
                state = flatten_rtmdp_obs(state, env.action_space.n)
            action = policy(state)
            state, reward, done, _ = env.step(action)
            cum_rew += reward

    return cum_rew / trials
