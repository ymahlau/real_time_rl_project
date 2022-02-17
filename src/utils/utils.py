import random
from collections import deque
from typing import Union, Any, Tuple, List, Callable

import gym
import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add_data(self, data: Tuple[Any, int, float, Any, bool]):  # state, action, reward, next_state, done
        self.buffer.append(data)

    def capacity_reached(self):
        return len(self.buffer) >= self.buffer.maxlen

    def sample(self, sample_size: int) -> List[Tuple[Any, int, float, Any, bool]]:
        return random.sample(self.buffer, sample_size)


@torch.no_grad()
def moving_average(target_params, current_params, factor):
    for t, c in zip(target_params, current_params):
        t += factor * (c - t)


def flatten_rtmdp_obs(obs: Union[np.ndarray, Tensor], num_actions: int) -> List[Any]:
    """
    Converts the observation tuple (s,a) returned by rtmdp
    into a single sequence s + one_hot_encoding(a)
    """
    return list(obs[0]) + list(one_hot_encoding(num_actions, obs[1]))


def one_hot_encoding(size: int, pos: int) -> np.ndarray:
    """
    Creates a one-hot-encoding given an array size and position
    """
    assert size > pos >= 0
    one_hot = np.zeros(size)
    one_hot[pos] = 1
    return one_hot
