from typing import Tuple, Optional

import torch
from torch import Tensor


class ReplayBuffer:

    def __init__(self, obs_len: int, capacity: int = 10000, seed: Optional[int] = None):
        self.obs_len = obs_len
        self.capacity = capacity

        self.next_idx = 0  # next index to insert new observation
        self.num_items = 0  # number of items in buffer

        self.obs = torch.zeros(size=(capacity, obs_len), dtype=torch.float)
        self.actions = torch.zeros(size=(capacity, 1), dtype=torch.int)
        self.rewards = torch.zeros(size=(capacity, 1), dtype=torch.float)
        self.next_states = torch.zeros(size=(capacity, obs_len), dtype=torch.float)
        self.dones = torch.zeros(size=(capacity, 1), dtype=torch.bool)

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self):
        return self.num_items

    def capacity_reached(self) -> bool:
        return self.num_items == self.capacity

    def add_data(self, data: Tuple[Tensor, int, float, Tensor, bool]):
        """
        data = (state, action, reward, next_state, done)
        All tensors have to be 1-dimensional
        """
        new_obs = data[0]
        new_action = torch.tensor(data[1], dtype=torch.int)
        new_reward = torch.tensor(data[2], dtype=torch.float)
        new_next_action = data[3]
        new_done = torch.tensor(data[4], dtype=torch.bool)

        self.obs[self.next_idx] = new_obs
        self.actions[self.next_idx] = new_action
        self.rewards[self.next_idx] = new_reward
        self.next_states[self.next_idx] = new_next_action
        self.dones[self.next_idx] = new_done

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.num_items = min(self.capacity, self.num_items + 1)

    def sample(self, sample_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sampled_idx = torch.randint(0, self.num_items, size=(sample_size,), generator=self.generator)

        sampled_obs = self.obs[sampled_idx]
        sampled_actions = self.actions[sampled_idx].squeeze(dim=1).long()
        sampled_rewards = self.rewards[sampled_idx].squeeze(dim=1)
        sampled_next_states = self.next_states[sampled_idx]
        sampled_dones = self.dones[sampled_idx].squeeze(dim=1).float()

        return sampled_obs, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones
