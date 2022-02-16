from typing import Tuple, Any, List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from src.agents import ActorCritic, PolicyValueNetwork
from src.utils.utils import moving_average


class RTAC(ActorCritic):

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
            double_value: bool = False,
            hidden_size: int = 256,
            num_layers: int = 2,
            target_smoothing_factor: float = 0.005,
            shared_parameters: bool = False):

        if not isinstance(env.observation_space, gym.spaces.Tuple) or len(env.observation_space) != 2:
            raise ValueError('RTAC needs a tuple with two entries as observations space in the given environment')
        if not isinstance(env.observation_space[1], gym.spaces.Discrete) \
                or not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError('RTAC needs discrete action space (as output and second entry in input tuple)')

        super().__init__(
            env,
            use_target=use_target,
            batch_size=batch_size,
            buffer_size=buffer_size,
            discount_factor=discount_factor,
            double_value = double_value,
            reward_scaling_factor = reward_scaling_factor,
        )

        # Scalar attributes
        self.entropy_scale = entropy_scale
        self.actor_critic_factor = actor_critic_factor
        self.target_smoothing_factor = target_smoothing_factor
        self.num_obs = len(env.observation_space[0].shape)

        # networks
        self.input_size = env.observation_space[0].shape[0] + env.observation_space[1].n
        self.network = PolicyValueNetwork(self.input_size, self.num_actions, num_layers=num_layers,
                                          hidden_size=hidden_size, shared_parameters=shared_parameters, double_value = self.double_value)
        if use_target:
            self.target_network = PolicyValueNetwork(self.input_size, self.num_actions, num_layers=num_layers,
                                                     hidden_size=hidden_size, shared_parameters=shared_parameters, double_value = self.double_value)

        # optimizer and loss
        self.optim = optim.Adam(self.network.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def load_network(self, checkpoint: str):
        """
            Loads the model with parameters contained in the files in the
            path checkpoint.

            checkpoint: Absolute path without ending to the two files the model is saved in.
        """
        self.network.load_state_dict(torch.load(f"{checkpoint}.model"))
        if self.use_target:
            self.target_network.load_state_dict(torch.load(f"{checkpoint}.model"))
        print(f"Continuing training on {checkpoint}.")

    def save_network(self, log_dest: str):
        """
           Saves the model with parameters to the files referred to by the file path log_dest.
           log_dest: Absolute path without ending to the two files the model is to be saved in.
       """
        torch.save(self.network.state_dict(), f"{log_dest}.model")
        print("Saved current training progress")

    def flatten_observation(self, obs: Tuple[Any, int]) -> Tensor:
        """
        Converts the observation tuple (s,a) returned by rtmdp
        into a single sequence s + one_hot_encoding(a)
        """
        last_state = torch.tensor(obs[0], dtype=torch.float)
        one_hot = F.one_hot(torch.tensor(obs[1]), num_classes=self.num_actions).float()

        flattened_obs = torch.cat((last_state, one_hot), dim=0)
        return flattened_obs

    def split_states(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        splits states x_t = (s_t, a_t)
        """
        return torch.split(states, [self.num_obs, self.num_actions], dim=1)

    def act(self, obs: Tuple[Any, int]) -> int:
        obs = self.flatten_observation(obs)
        action = self.network.act(obs)
        return action

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        obs = self.flatten_observation(obs)
        value = self.network.get_value(obs)
        return value

    def get_action_distribution(self, obs: Tuple[Any, int]) -> Tensor:
        obs = self.flatten_observation(obs)
        distribution = self.network.get_action_distribution(obs)
        return distribution

    def update(self, samples: List[Tuple[Tuple[Any, int], int, float, Tuple[Any, int], bool]]):
        self.optim.zero_grad()

        # states x_t = (s_t, a_t), obs is the concatenated state and value
        current_obs = torch.stack([self.flatten_observation(samples[i][0]) for i in range(self.batch_size)], dim=0)
        current_state, current_action = self.split_states(current_obs)  # s_t, a_t
        dist_current_obs = self.network.get_action_distribution(current_obs)  # pi(a | s_t, a_t)

        # reward tensor
        rewards = torch.tensor([samples[i][2] for i in range(self.batch_size)], dtype=torch.float)  # r(s_t, a_t)

        # tensor of next states: x_t+1 = (s_t+1, a_t+1)
        next_obs = torch.stack([self.flatten_observation(samples[i][3]) for i in range(self.batch_size)], dim=0)
        next_state, next_action = self.split_states(next_obs)  # s_t+1, a_t+1

        # done values
        dones = torch.tensor([samples[i][4] for i in range(self.batch_size)], dtype=torch.float)
        dones_expanded = dones[:, None].expand(-1, self.num_actions)

        # (s_t+1, a_t) for all a_t, shape (batch, num_act, num_states)
        next_state_expanded = next_state[:, None, :].expand(-1, self.num_actions, -1)
        # shape = (batch, num_act, num_act)
        all_actions_expanded = torch.eye(self.num_actions)[None, :, :].expand(self.batch_size, -1, -1)

        next_obs_all_actions = torch.cat((next_state_expanded, all_actions_expanded), dim=2)
        flattened = torch.flatten(next_obs_all_actions, start_dim=0, end_dim=1)  # (batch * act, num_states + num_act)

        if self.use_target:
            values_flattened = self.target_network.get_value(flattened)  # v(s_t+1, a_t)
            values = self.target_network.get_value(current_obs)  # v(s_t, a_t)
        else:
            values_flattened = self.network.get_value(flattened)
            values = self.network.get_value(current_obs)
        values_unflattened = values_flattened.reshape(self.batch_size, self.num_actions)

        # expectation of next state values
        value_expectation = (1 - dones_expanded) * self.discount_factor * values_unflattened
        value_expectation -= self.entropy_scale * dist_current_obs.log()
        value_expectation = torch.sum(dist_current_obs * value_expectation, dim=1)

        targets = (rewards + value_expectation).detach().float()
        value_loss = self.mse_loss(values.squeeze(dim=1), targets)

        # compute policy loss
        value_next_obs = self.network.get_value(flattened).reshape(self.batch_size, self.num_actions)

        critic_approx = (1 - dones_expanded) * self.discount_factor * (1 / self.entropy_scale)
        critic_approx = (critic_approx * value_next_obs).detach()

        kl_div = torch.sum(dist_current_obs * (dist_current_obs.log() - critic_approx), dim=1)
        policy_loss = kl_div.mean()

        # Update parameters
        loss = self.actor_critic_factor * policy_loss + value_loss
        loss.backward()
        self.optim.step()

        if self.use_target:
            moving_average(self.target_network.parameters(), self.network.parameters(),
                           self.target_smoothing_factor)
