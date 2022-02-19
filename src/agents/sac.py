from typing import Tuple, Any, Optional

import gym
import torch
import torch.nn.functional as F
from torch import Tensor

from src.agents import ActorCritic


class SAC(ActorCritic):
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
            use_device: bool = True,
            seed: Optional[int] = None,
    ):
        # arguments for network generation
        if network_kwargs is None:
            network_kwargs = {}
        network_kwargs['value_input_size'] = env.observation_space.shape[0] + env.action_space.n
        network_kwargs['policy_input_size'] = env.observation_space.shape[0]
        network_kwargs['output_size'] = env.action_space.n

        num_obs = env.observation_space.shape[0]

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

        self.entropy_scale = entropy_scale

    def obs_to_tensor(self, obs: Any) -> Tensor:
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        return obs_tensor

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        action_tensor = torch.tensor(obs[1])
        one_hot_action = F.one_hot(action_tensor, num_classes=self.num_actions).to(self.device)
        obs_tensor = self.obs_to_tensor(obs[0])

        concat_tensor = torch.cat((obs_tensor, one_hot_action), dim=0)
        value = self.network.get_value(concat_tensor)
        return value

    def value_loss(self, samples: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        states = samples[0]
        actions = samples[1]
        rewards = samples[2]
        next_states = samples[3]
        dones = samples[4]

        # convert actions to one hot
        actions_one_hot = F.one_hot(actions, num_classes=self.num_actions).to(self.device)

        dones_expanded = dones[:, None].expand(-1, self.num_actions)
        next_actions_dist = self.network.get_action_distribution(next_states)

        flattened = self.all_state_action_pairs(next_states)

        # prediction of target q-values
        if self.use_target:
            targets_value = self.target.get_value(flattened)
        else:
            targets_value = self.network.get_value(flattened)
        values_unflattened = targets_value.unflatten(dim=0, sizes=(self.batch_size, self.num_actions)).squeeze(dim=2)

        # target has to be unnormalized to add to new reward and compute new statistics
        if self.network.normalized:
            if self.use_target:
                values_unflattened = self.target.unnormalize(values_unflattened)
            else:
                values_unflattened = self.network.unnormalize(values_unflattened)

        # compute new targets
        targets_discount = self.discount_factor * (1 - dones_expanded) * values_unflattened
        targets_entropy = self.entropy_scale * next_actions_dist.log()

        target_expectation = torch.sum(next_actions_dist * (targets_discount - targets_entropy), dim=1).float()
        targets = (rewards + target_expectation).detach().float()

        if self.network.normalized:
            targets = self.handle_normalization(targets)

        # compute value predictions
        state_action_pairs = torch.cat((states, actions_one_hot), dim=1).float()
        values = self.network.get_value(state_action_pairs).squeeze(dim=1)

        loss = self.mse_loss(values, targets)
        return loss

    def policy_loss(self, samples: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        states = samples[0]
        action_dist = self.network.get_action_distribution(states)

        flattened = self.all_state_action_pairs(states)
        values = self.network.get_value(flattened)
        values_unflattened = values.unflatten(dim=0, sizes=(self.batch_size, self.num_actions)).squeeze(dim=2)

        if self.network.normalized:
            values_unflattened = self.network.unnormalize(values_unflattened)

        values_unflattened = values_unflattened.detach().float()

        kl_div_term = action_dist.log() - self.discount_factor * (1 / self.entropy_scale) * values_unflattened
        policy_loss = torch.sum(action_dist * kl_div_term, dim=1)
        if self.network.normalized:
            policy_loss = self.network.normalize(policy_loss)

        loss = policy_loss.mean()
        return loss
