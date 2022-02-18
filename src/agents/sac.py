from functools import partial
from typing import List, Tuple, Any, Optional

import gym
import torch
from torch import Tensor

import torch.nn as nn
from src.agents import ActorCritic
from src.agents.networks import PolicyValueNetwork
from src.utils.utils import moving_average, one_hot_encoding


class SAC(ActorCritic):
    def __init__(
            self,
            env: gym.Env,
            eval_env: Optional[gym.Env] = None,
            entropy_scale: float = 0.2,
            discount_factor: float = 0.99,
            reward_scaling_factor: float = 1.0,
            lr: float = 0.0003,
            actor_critic_factor: float = 0.1,
            buffer_size: int = 10000,
            batch_size: int = 256,
            use_target: bool = False,
            double_value: bool = False,
            hidden_size: int = 256,
            num_layers: int = 2,
            target_smoothing_factor: float = 0.005,
            normalized: bool = False,
            pop_art_factor: float = 0.0003,
            epsilon: float = 1e-6  # for numerical stability, not given in rtac-paper
    ):
        super().__init__(
            env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            use_target=use_target,
            double_value=double_value,
            normalized=normalized,
            batch_size=batch_size,
            discount_factor=discount_factor,
            reward_scaling_factor=reward_scaling_factor,
        )

        # scalar
        self.entropy_scale = entropy_scale
        self.lr = lr
        self.actor_critic_factor = actor_critic_factor
        self.target_smoothing_factor = target_smoothing_factor
        self.normalized = normalized
        self.pop_art_factor = pop_art_factor

        self.value_input_size = self.env.observation_space.shape[0] + self.num_actions
        self.policy_input_size = self.env.observation_space.shape[0]

        # networks
        self.network = PolicyValueNetwork(
            self.value_input_size,
            self.policy_input_size,
            self.num_actions,
            shared_parameters=False,
            double_value=double_value,
            hidden_size=hidden_size,
            num_layers=num_layers,
            normalized=normalized,
            pop_art_factor=pop_art_factor,
            epsilon=epsilon
        )

        if self.use_target:
            self.target = PolicyValueNetwork(
                self.value_input_size,
                self.policy_input_size,
                self.num_actions,
                shared_parameters=False,
                double_value=double_value,
                hidden_size=hidden_size,
                num_layers=num_layers,
                normalized=normalized,
                pop_art_factor=pop_art_factor,
                epsilon=epsilon
            )

        # optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        # self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr * actor_critic_factor)

        # functions
        self.mse_loss = nn.MSELoss()
        self.one_hot = partial(one_hot_encoding, self.num_actions)

    def load_network(self, checkpoint: str):
        """
            Loads the model with parameters contained in the files in the
            path checkpoint.

            checkpoint: Absolute path without ending to the two files the model is saved in.
        """
        self.network.load_state_dict(torch.load(f"{checkpoint}.model"))
        # self.value.load_state_dict(torch.load(f"{checkpoint}.val_model"))
        if self.use_target:
            self.target.load_state_dict(torch.load(f"{checkpoint}.model"))
        print(f"Continuing training on {checkpoint}.")

    def save_network(self, log_dest: str):
        """
           Saves the model with parameters to the files referred to by the file path log_dest.
           log_dest: Absolute path without ending to the two files the model is to be saved in.
       """
        torch.save(self.network.state_dict(), f"{log_dest}.pol_model")
        # torch.save(self.value.state_dict(), f"{log_dest}.val_model")
        print(f"Saved current training progress to {log_dest}")

    def act(self, obs: Any) -> int:
        action = self.network.act(torch.tensor(obs))
        return action

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        obs_tensor = torch.cat((torch.tensor(obs[0]), torch.tensor(self.one_hot(obs[1]))), dim=0)
        value = self.network.get_value(obs_tensor)
        return value

    def get_action_distribution(self, obs: Any) -> Tensor:
        obs_tensor = torch.tensor(obs)
        dist = self.network.get_action_distribution(obs_tensor)
        return dist

    def value_loss(
            self,
            states: Tensor,
            actions: Tensor,
            rewards: Tensor,
            next_states: Tensor,
            dones: Tensor) -> Tensor:

        dones_expanded = dones.expand(-1, self.num_actions)
        next_actions_dist = self.network.get_action_distribution(next_states)
        next_actions = [torch.tensor(self.one_hot(a)) for a in range(self.num_actions)]

        # prediction of target q-values
        if self.use_target:
            targets_value = [self.target.get_value(torch.cat((next_states, next_actions[a].expand(self.batch_size, -1)), dim=1))
                             for a in range(self.num_actions)]
        else:
            targets_value = [self.network.get_value(
                torch.cat((next_states, next_actions[a].expand(self.batch_size, -1)), dim=1))
                for a in range(self.num_actions)]

        targets_value = torch.squeeze(torch.stack(targets_value, dim=1), dim=2).float()

        # target has to be unnormalized to add to new reward and compute new statistics
        if self.normalized:
            if self.use_target:
                targets_value = self.target.unnormalize(targets_value)
            else:
                targets_value = self.network.unnormalize(targets_value)

        # compute new targets
        targets_discount = self.discount_factor * (1 - dones_expanded) * targets_value
        targets_entropy = self.entropy_scale * next_actions_dist.log()

        targets = torch.sum(next_actions_dist * (targets_discount - targets_entropy), dim=1, dtype=torch.float)
        targets = rewards + targets.unsqueeze(1).detach().float()

        # update normalization parameters
        if self.normalized:
            self.network.update_normalization(targets)
            if self.use_target:
                self.target.update_normalization(targets)

        # compute normalized loss
        if self.normalized:
            if self.use_target:
                norm_target = self.target.normalize(targets)
            else:
                norm_target = self.network.normalize(targets)
            targets = norm_target.detach().float()

        values = self.network.get_value(torch.cat((states, actions), dim=1)).float()

        loss = self.mse_loss(values, targets)
        return loss

    def policy_loss(self, states: Tensor) -> Tensor:
        next_actions_dist = self.network.get_action_distribution(states)
        values = [self.network.get_value(torch.cat((states, torch.tensor(self.one_hot(a)).expand(self.batch_size, -1)), dim=1))
                  for a in range(self.num_actions)]
        values = torch.squeeze(torch.stack(values, dim=1), dim=2).detach()
        if self.normalized:
            values = self.network.unnormalize(values)

        kl_div_term = next_actions_dist.log() - self.discount_factor * (1 / self.entropy_scale) * values
        policy_loss = torch.sum(next_actions_dist * kl_div_term, dim=1)
        if self.normalized:
            policy_loss = self.network.normalize(policy_loss)

        loss = policy_loss.mean()
        return loss

    def update(self, samples: List[Tuple[Any, int, float, Any, bool]]):
        state_batch = torch.tensor([s[0] for s in samples]).float()
        action_batch = torch.tensor([self.one_hot(s[1]) for s in samples]).float()
        reward_batch = torch.tensor([s[2] for s in samples]).unsqueeze(dim=1).float()
        next_state_batch = torch.tensor([s[3] for s in samples]).float()
        done_batch = torch.tensor([s[4] for s in samples], dtype=torch.float).unsqueeze(dim=1)

        self.optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optim.step()

        value_loss = self.value_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        policy_loss = self.policy_loss(state_batch)

        loss = self.actor_critic_factor * policy_loss + value_loss
        # self.policy_optim.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target:
            moving_average(self.target.parameters(), self.network.parameters(),
                           self.target_smoothing_factor)
