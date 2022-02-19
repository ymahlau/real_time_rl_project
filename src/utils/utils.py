from typing import Union, Any, List

import numpy as np
import torch
from torch import Tensor


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
