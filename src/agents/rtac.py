from typing import Tuple, Any, List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from src.agents import ActorCritic


class RTAC(ActorCritic):

    def __init__(
            self,
            env: gym.Env,
            entropy_scale: float = 0.2,
            actor_critic_factor: float = 0.1,
            buffer_size: int = 10000,
            batch_size: int = 256,
            lr: float = 0.0003,
            use_target: bool = False,
            discount_factor: float = 0.99,
            hidden_size: int = 256,
            target_smoothing_factor: float = 0.005):

        if not isinstance(env.observation_space, gym.spaces.Tuple) or len(env.observation_space) != 2:
            raise ValueError('RTAC needs a tuple with two entries as observations space in the given environment')
        if not isinstance(env.observation_space[1], gym.spaces.Discrete) \
                or not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError('RTAC needs discrete action space (as output and second entry in input tuple)')

        input_size = env.observation_space[0].shape[0] + env.observation_space[1].n
        super().__init__(
            env,
            input_size=input_size,
            use_target=use_target,
            batch_size=batch_size,
            buffer_size=buffer_size,
            discount_factor=discount_factor,
            hidden_size=hidden_size,
        )

        # Scalar attributes
        self.entropy_scale = entropy_scale
        self.actor_critic_factor = actor_critic_factor
        self.target_smoothing_factor = target_smoothing_factor
        self.num_obs = len(env.observation_space[0].shape)

        # optimizer and loss
        self.optim = optim.Adam(self.network.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

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

    def act(self, obs: Tuple[Any, int]) -> Tensor:
        obs = self.flatten_observation(obs)
        action = self.network.act(obs)
        return action

    def get_value(self, obs: Tuple[Any, int]) -> Tensor:
        obs = self.flatten_observation(obs)
        value = self.network.get_value(obs)
        return value

    def get_action_distribution(self, obs: Tuple[Any, int]):
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
        critic_approx *= value_next_obs.detach()

        kl_div = torch.sum(dist_current_obs * (dist_current_obs.log() - critic_approx), dim=1)
        policy_loss = kl_div.mean()

        # Update parameters
        loss = self.actor_critic_factor * policy_loss + (1 - self.actor_critic_factor) * value_loss
        loss.backward()
        self.optim.step()

    # def train_old(
    #         self,
    #         checkpoint: Optional[str] = None,
    #         log_dest: Optional[str] = None,
    #         log_progress: bool = False,
    #         logging_rate: int = 30 * 60,
    #         training_time: Optional[float] = None) -> None:
    #     """
    #         Trains the RTAC model with the real-time-soft-actor-critic-algorithm.
    #
    #         checkpoint: An absolute path without ending to two files where a model is saved
    #         log_dest: An absolute path without ending where to trained model is to regularly saved in
    #         log_progress: Whether the training progress is supposed to be logged
    #         logging_rate: The rate at which the model is regularly saved given in seconds
    #         training_time: If set, training will terminate after training_time seconds
    #     """
    #     if checkpoint is not None:
    #         self.retrieve_model(checkpoint)
    #
    #     last_log = time.time()
    #     start_time = time.time()
    #     env_steps = 0
    #     while True:
    #
    #         done = False
    #         state = self.env.reset()
    #
    #         while not done:
    #
    #             if training_time is not None and time.time() - start_time > training_time:
    #                 if log_progress:
    #                     self.save_model(log_dest)
    #                 return
    #
    #             # If necessary, log model
    #             if log_progress and (time.time() - last_log > logging_rate):
    #                 self.save_model(log_dest)
    #                 last_log = time.time()
    #
    #             # Perform step on env and add step data to replay buffer
    #             state = flatten_rtmdp_obs(state, self.nom_actions)
    #             action = self.network.act(state)
    #             next_state, reward, done, _ = self.env.step(action)
    #             self.replay.add_data((torch.tensor(state), reward, torch.tensor(next_state[0]), done))
    #             state = next_state
    #
    #             env_steps += 1
    #             if env_steps % 100 == 0 and self.eval_env is not None and self.replay.capacity_reached():
    #                 avg_rew = evaluate_policy(self.network.act, self.eval_env)
    #                 print(f"Policy gathered a average reward of {avg_rew}")
    #
    #             # Perform single update step
    #             if self.replay.capacity_reached():
    #                 """
    #                 Retrieve relevant tensors from the sample batch
    #                 """
    #                 sample = np.array(self.replay.sample(self.batch_size), dtype=object)
    #                 states = torch.stack(list(sample[:, 0]), dim=0)  # tensor of states
    #                 distributions = self.network.get_action_distribution(
    #                     states)  # tensor of action distributions on states
    #                 rewards = torch.tensor(np.array(sample[:, 1], dtype=float))  # reward tensor
    #                 next_states_obs = torch.stack(list(sample[:, 2]), dim=0)  # tensor of next states without the action
    #                 dones = torch.tensor(np.array(sample[:, 3], dtype=float))
    #
    #                 values_next_states = []
    #                 for j in range(self.nom_actions):
    #                     values_next_states.append(self.network.get_value(torch.tensor(
    #                         [flatten_rtmdp_obs((list(next_states_obs[i]), j), self.nom_actions) for i in
    #                          range(self.batch_size)])))
    #                 values_next_states = torch.squeeze(torch.stack(values_next_states, dim=1),
    #                                                    dim=2)  # tensor of values of next states with all actions
    #
    #                 if self.use_target:
    #                     values_next_states_target = []
    #                     for j in range(self.nom_actions):
    #                         values_next_states_target.append(self.target.get_value(torch.tensor(
    #                             [flatten_rtmdp_obs((list(next_states_obs[i]), j), self.nom_actions) for i in
    #                              range(self.batch_size)])))
    #                     values_next_states_target = torch.squeeze(torch.stack(values_next_states_target, dim=1),
    #                                                               dim=2)  # tensor of values of next states with all actions
    #
    #                 # Calculate loss functions now
    #                 # calc value func loss
    #                 v = values_next_states_target if self.use_target else values_next_states
    #                 targets = (rewards + torch.sum(distributions * (
    #                         (1 - dones[:, None]) * self.discount * v - self.entropy_scale * distributions.log()),
    #                                                1)).detach().float()
    #                 values = torch.squeeze(self.network.get_value(states), 1)
    #                 loss_value = self.mse_loss(values, targets)
    #
    #                 # calc policy func loss
    #                 loss_pol = torch.sum(distributions * (distributions.log() - (1 - dones[:, None]) * self.discount * (
    #                         1 / self.entropy_scale) * (values_next_states.detach())), 1).mean()
    #
    #                 loss = self.actor_critic_factor * loss_pol + (1 - self.actor_critic_factor) * loss_value
    #
    #                 # Update parameters
    #                 self.optim.zero_grad()
    #                 loss.backward()
    #                 self.optim.step()
    #
    #                 if self.use_target:
    #                     moving_average(self.target.parameters(), self.network.parameters(),
    #                                    self.target_smoothing_factor)
