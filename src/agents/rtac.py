from typing import Tuple, Any, Optional

import gym
import torch
import torch.nn.functional as F
from torch import Tensor

from src.agents import ActorCritic


class RTAC(ActorCritic):

    def __init__(
            self,
            env: gym.Env,
            network_kwargs: Optional[dict[str, Any]] = None,
            eval_env: Optional[gym.Env] = None,
            entropy_scale: float = 0.2,
            discount_factor: float = 0.99,
            reward_scaling_factor: float = 1.0,
            lr: float = 0.0003,
            actor_critic_factor: float = 0.1,
            buffer_size: int = 10000,
            batch_size: int = 256,
            use_target: bool = False,
            target_smoothing_factor: float = 0.005,
            use_device: bool = False,
            seed: Optional[int] = None,
    ):

        if not isinstance(env.observation_space, gym.spaces.Tuple) or len(env.observation_space) != 2:
            raise ValueError('RTAC needs a tuple with two entries as observations space in the given environment')

        if not isinstance(env.observation_space[1], gym.spaces.Discrete) \
                or not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError('RTAC needs discrete action space (as output and second entry in input tuple)')

        # arguments for network generation
        if network_kwargs is None:
            network_kwargs = {}
        network_kwargs['value_input_size'] = env.observation_space[0].shape[0] + env.observation_space[1].n
        network_kwargs['policy_input_size'] = env.observation_space[0].shape[0] + env.observation_space[1].n
        network_kwargs['output_size'] = env.action_space.n

        num_obs = env.observation_space[0].shape[0]

        super().__init__(
            env=env,
            num_obs=num_obs,
            network_kwargs=network_kwargs,
            eval_env=eval_env,
            buffer_size=buffer_size,
            use_target=use_target,
            batch_size=batch_size,
            discount_factor=discount_factor,
            reward_scaling_factor=reward_scaling_factor,
            lr=lr,
            actor_critic_factor=actor_critic_factor,
            target_smoothing_factor=target_smoothing_factor,
            use_device=use_device,
            seed=seed,
        )

        # Scalar attributes
        self.entropy_scale = entropy_scale

    def split_states(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        splits states x_t = (s_t, a_t)
        """
        return torch.split(states, [self.num_obs, self.num_actions], dim=1)

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        obs = self.obs_to_tensor(obs)
        value = self.network.get_value(obs)
        return value

    def obs_to_tensor(self, obs: Tuple[Any, int]) -> Tensor:
        """
        Converts the observation tuple (s,a) returned by rtmdp
        into a single sequence s + one_hot_encoding(a)
        """
        last_state = torch.tensor(obs[0], dtype=torch.float, device=self.device)
        one_hot = F.one_hot(torch.tensor(obs[1], device=self.device), num_classes=self.num_actions).float()

        flattened_obs = torch.cat((last_state, one_hot), dim=0)
        return flattened_obs

    def value_loss(self, samples: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        # states x_t = (s_t, a_t), obs is the concatenated state and value
        current_obs = samples[0]
        dist_current_obs = self.network.get_action_distribution(current_obs)  # pi(a | s_t, a_t)
        dist_current_obs_clamped = torch.clamp(dist_current_obs, min=1e-8)
        dist_current_obs_log_clamped = dist_current_obs_clamped.log()
        rewards = samples[2]  # r(s_t, a_t)

        # tensor of next states: x_t+1 = (s_t+1, a_t+1)
        next_obs = samples[3]
        next_state, next_action = self.split_states(next_obs)  # s_t+1, a_t+1

        # done values
        dones = samples[4]
        dones_expanded = dones[:, None].expand(-1, self.num_actions)

        flattened = self.all_state_action_pairs(next_state)

        if self.use_target:
            values_flattened = self.target.get_value(flattened)  # v(s_t+1, a_t)
        else:
            values_flattened = self.network.get_value(flattened)
        values_unflattened = values_flattened.reshape(self.batch_size, self.num_actions)

        # target has to be unnormalized to add to new reward and compute new statistics
        if self.network.normalized:
            if self.use_target:
                values_unflattened = self.target.unnormalize(values_unflattened)
            else:
                values_unflattened = self.network.unnormalize(values_unflattened)

        # expectation of next state values
        value_expectation = (1 - dones_expanded) * self.discount_factor * values_unflattened
        value_expectation -= self.entropy_scale * dist_current_obs_log_clamped
        value_expectation = torch.sum(dist_current_obs * value_expectation, dim=1)
        targets = (rewards + value_expectation).detach().float()

        if self.network.normalized:
            targets = self.handle_normalization(targets)

        values = self.network.get_value(current_obs).squeeze(dim=1)
        value_loss = self.mse_loss(values, targets)

        return value_loss

    def policy_loss(self, samples: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        # states x_t = (s_t, a_t), obs is the concatenated state and value
        current_obs = samples[0]
        dist_current_obs = self.network.get_action_distribution(current_obs)  # pi(a | s_t, a_t)
        dist_current_obs_clamped = torch.clamp(dist_current_obs, min=1e-8)
        dist_current_obs_log_clamped = dist_current_obs_clamped.log()

        # done values
        dones = samples[4]
        dones_expanded = dones[:, None].expand(-1, self.num_actions)

        # tensor of next states: x_t+1 = (s_t+1, a_t+1)
        next_obs = samples[3]
        next_state, next_action = self.split_states(next_obs)  # s_t+1, a_t+1

        flattened = self.all_state_action_pairs(next_state)

        value_next_obs = self.network.get_value(flattened).reshape(self.batch_size, self.num_actions)

        if self.network.normalized:
            value_next_obs = self.network.unnormalize(value_next_obs)

        critic_approx = (1 - dones_expanded) * self.discount_factor * (1 / self.entropy_scale)
        critic_approx = (critic_approx * value_next_obs).detach()

        kl_div = torch.sum(dist_current_obs * (dist_current_obs_log_clamped - critic_approx), dim=1)
        if self.network.normalized:
            kl_div = self.network.normalize(kl_div)

        policy_loss = kl_div.mean()

        return policy_loss
