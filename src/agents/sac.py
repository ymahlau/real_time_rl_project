from typing import List, Tuple, Any

import gym
import torch
from torch import Tensor

from src.agents import ActorCritic
from src.agents.networks import PolicyNetwork, ValueNetwork
from src.utils.utils import ReplayBuffer, evaluate_policy, moving_average
import numpy as np


class SAC(ActorCritic):
    def __init__(
            self,
            env: gym.Env,
            entropy_scale: float = 0.2,
            discount_factor: float = 0.99,
            reward_scaling_factor: float = 1.0,
            lr: float = 0.0003,
            actor_critic_factor: float = 0.1,
            buffer_size: int = 10000,
            batch_size: int = 256,
            use_target: bool = False,
            double_target: bool = False,
            hidden_size: int = 256,
            num_layers: int = 2,
            target_smoothing_factor: float = 0.005):
        super().__init__(
            env,
            buffer_size=buffer_size,
            use_target=use_target,
            double_target = double_target,
            batch_size=batch_size,
            discount_factor=discount_factor,
            reward_scaling_factor = reward_scaling_factor,
        )

        # scalar
        self.entropy_scale = entropy_scale
        self.lr = lr
        self.actor_critic_factor = actor_critic_factor
        self.num_actions = self.env.action_space.n
        self.target_smoothing_factor = target_smoothing_factor

        # networks
        self.value = ValueNetwork(self.env.observation_space.shape[0] + 1,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers)
        if self.use_target:
            self.target = ValueNetwork(self.env.observation_space.shape[0] + 1,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers)
        self.policy = PolicyNetwork(self.env.observation_space.shape[0],
                                    self.env.action_space.n,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers)

        # optimizer
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr * actor_critic_factor)

    def load_network(self, checkpoint: str):
        """
            Loads the model with parameters contained in the files in the
            path checkpoint.

            checkpoint: Absolute path without ending to the two files the model is saved in.
        """
        self.policy.load_state_dict(torch.load(f"{checkpoint}.pol_model"))
        self.value.load_state_dict(torch.load(f"{checkpoint}.val_model"))
        if self.use_target:
            self.target.load_state_dict(torch.load(f"{checkpoint}.val_model"))
        print(f"Continuing training on {checkpoint}.")

    def save_network(self, log_dest: str):
        """
           Saves the model with parameters to the files referred to by the file path log_dest.
           log_dest: Absolute path without ending to the two files the model is to be saved in.
       """
        torch.save(self.policy.state_dict(), f"{log_dest}.pol_model")
        torch.save(self.value.state_dict(), f"{log_dest}.val_model")
        print("Saved current training progress")

    def act(self, obs: Any) -> int:
        action = self.policy.act(torch.tensor(obs))
        return action

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        obs_tensor = torch.cat((torch.tensor(obs[0]), torch.tensor([obs[1]])), dim=0)
        value = self.value(obs_tensor)
        return value

    def get_action_distribution(self, obs: Any) -> Tensor:
        obs_tensor = torch.tensor(obs)
        dist = self.policy.get_action_distribution(obs_tensor)
        return dist

    def value_loss(
            self,
            states: Tensor,
            actions: Tensor,
            rewards: Tensor,
            next_states: Tensor,
            dones: Tensor) -> Tensor:
        next_actions_dist = self.policy.get_action_distribution(next_states)
        value_targets = [self.value(torch.cat((next_states, (torch.ones(self.batch_size)[:, None] * a)), dim=1))
                         for a in range(self.num_actions)]
        value_targets = torch.squeeze(torch.stack(value_targets, dim=1), dim=2)

        targets = rewards + torch.sum(next_actions_dist * (self.discount_factor * (1 - dones.expand(-1, self.num_actions)) *
                  value_targets - self.entropy_scale * next_actions_dist.log()), 1).unsqueeze(1).detach()
        values = self.value(torch.cat((states, actions), dim=1))

        return torch.pow(values - targets, 2).mean()

    def policy_loss(self, states: Tensor, done_batch: Tensor) -> Tensor:
        next_actions_dist = self.policy.get_action_distribution(states)
        values = [self.value(torch.cat((states, (torch.ones(self.batch_size)[:, None] * a)), dim=1)) for a in
                  range(self.num_actions)]
        values = torch.squeeze(torch.stack(values, dim=1), dim=2).detach()

        kl_div_term = next_actions_dist.log() - self.discount_factor * (1 / self.entropy_scale) * values
        policy_loss = torch.sum(next_actions_dist * kl_div_term, dim=1).mean()
        return policy_loss

    def update(self, samples: List[Tuple[Any, int, float, Any, bool]]):
        state_batch = torch.tensor([s[0] for s in samples])
        action_batch = torch.tensor([s[1] for s in samples]).unsqueeze(dim=1)
        reward_batch = torch.tensor([s[2] for s in samples]).unsqueeze(dim=1)
        next_state_batch = torch.tensor([s[3] for s in samples])
        done_batch = torch.tensor([s[4] for s in samples], dtype=torch.float).unsqueeze(dim=1)

        value_loss = self.value_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        policy_loss = self.policy_loss(state_batch, done_batch)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.use_target:
            moving_average(self.target.parameters(), self.value.parameters(),
                           self.target_smoothing_factor)

