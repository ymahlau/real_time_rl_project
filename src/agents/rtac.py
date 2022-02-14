import time
from typing import Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.network import PolicyValueNetwork
from src.utils.utils import ReplayBuffer
from src.utils.utils import evaluate_policy, flatten_rtmdp_obs, moving_average


class RTAC:

    def __init__(
            self,
            env: gym.Env,
            eval_env: bool = None,
            entropy_scale: float = 0.2,
            discount: float = 0.99,
            lr: float = 0.0003,
            actor_critic_factor: float = 0.1,
            buffer_size: int = 10000,
            batch_size: int = 256,
            hidden_size: int = 256,
            num_hidden: int = 2,
            shared_parameters: bool = True,
            use_target: int = True,
            target_smoothing_factor: float = 0.005):

        # Scalar attributes
        self.entropy_scale = entropy_scale
        self.actor_critic_factor = actor_critic_factor
        self.discount = discount
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_smoothing_factor = target_smoothing_factor
        self.use_target = use_target

        # Env
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")

        self.env = env
        self.eval_env = eval_env
        self.nom_actions = env.action_space.n

        # Network
        self.shared_parameters = shared_parameters
        input_size = env.observation_space[0].shape[0] + env.action_space.n
        self.network = PolicyValueNetwork(input_size,
                                          env.action_space.n,
                                          hidden_size=hidden_size,
                                          num_hidden=num_hidden,
                                          shared_parameters=shared_parameters)

        if use_target:
            self.target = PolicyValueNetwork(input_size,
                                             env.action_space.n,
                                             hidden_size=hidden_size,
                                             num_hidden=num_hidden,
                                             shared_parameters=shared_parameters)

        self.optim = optim.Adam(self.network.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

        self.replay = ReplayBuffer(buffer_size)

    """
        Loads the model with parameters contained in the files in the 
        path checkpoint.
        
        checkpoint: Absolute path without ending to the two files the model is saved in.
    """

    def retrieve_model(self, checkpoint: str):
        self.network.load_state_dict(torch.load(f"{checkpoint}.model"))
        if self.use_target:
            self.target.load_state_dict(torch.load(f"{checkpoint}.model"))
        print(f"Continuing training on {checkpoint}.")

    """
        Saves the model with parameters to the files referred to by the file path log_dest.
        
        log_dest: Absolute path without ending to the two files the model is to be saved in.
    """

    def save_model(self, log_dest: str):
        torch.save(self.network.state_dict(), f"{log_dest}.model")
        print("Saved current training progress")

    """
        Trains the RTAC model with the real-time-soft-actor-critic-algorithm.
        
        checkpoint: An absolute path without ending to two files where a model is saved
        log_dest: An absolute path without ending where to trained model is to regularly saved in
        log_progress: Whether the training progress is supposed to be logged
        logging_rate: The rate at which the model is regularly saved given in seconds
        training_time: If set, training will terminate after training_time seconds
    """

    def train(
            self,
            checkpoint: Optional[str] = None,
            log_dest: Optional[str] = None,
            log_progress: bool = False,
            logging_rate: int = 30 * 60,
            training_time: Optional[float] = None) -> None:

        if checkpoint is not None:
            self.retrieve_model(checkpoint)

        last_log = time.time()
        start_time = time.time()
        env_steps = 0
        while True:

            done = False
            state = self.env.reset()

            while not done:

                if training_time is not None and time.time() - start_time > training_time:
                    if log_progress:
                        self.save_model(log_dest)
                    return

                # If necessary, log model
                if log_progress and (time.time() - last_log > logging_rate):
                    self.save_model(log_dest)
                    last_log = time.time()

                # Perform step on env and add step data to replay buffer
                state = flatten_rtmdp_obs(state, self.nom_actions)
                action = self.network.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay.add_data((torch.tensor(state), reward, torch.tensor(next_state[0]), done))
                state = next_state

                env_steps += 1
                if env_steps % 100 == 0 and self.eval_env is not None and self.replay.capacity_reached():
                    avg_rew = evaluate_policy(self.network.act, self.eval_env)
                    print(f"Policy gathered a average reward of {avg_rew}")

                # Perform single update step
                if self.replay.capacity_reached():
                    """
                    Retrieve relevant tensors from the sample batch
                    """
                    sample = np.array(self.replay.sample(self.batch_size), dtype=object)
                    states = torch.stack(list(sample[:, 0]), dim=0)  # tensor of states
                    distributions = self.network.get_action_distribution(
                        states)  # tensor of action distributions on states
                    rewards = torch.tensor(np.array(sample[:, 1], dtype=float))  # reward tensor
                    next_states_obs = torch.stack(list(sample[:, 2]), dim=0)  # tensor of next states without the action
                    dones = torch.tensor(np.array(sample[:, 3], dtype=float))

                    values_next_states = []
                    for j in range(self.nom_actions):
                        values_next_states.append(self.network.get_value(torch.tensor(
                            [flatten_rtmdp_obs((list(next_states_obs[i]), j), self.nom_actions) for i in
                             range(self.batch_size)])))
                    values_next_states = torch.squeeze(torch.stack(values_next_states, dim=1),
                                                       dim=2)  # tensor of values of next states with all actions

                    if self.use_target:
                        values_next_states_target = []
                        for j in range(self.nom_actions):
                            values_next_states_target.append(self.target.get_value(torch.tensor(
                                [flatten_rtmdp_obs((list(next_states_obs[i]), j), self.nom_actions) for i in
                                 range(self.batch_size)])))
                        values_next_states_target = torch.squeeze(torch.stack(values_next_states_target, dim=1),
                                                                  dim=2)  # tensor of values of next states with all actions

                    # Calculate loss functions now
                    # calc value func loss
                    v = values_next_states_target if self.use_target else values_next_states
                    targets = (rewards + torch.sum(distributions * (
                                (1 - dones[:, None]) * self.discount * v - self.entropy_scale * distributions.log()),
                                                   1)).detach().float()
                    values = torch.squeeze(self.network.get_value(states), 1)
                    loss_value = self.mse_loss(values, targets)

                    # calc policy func loss
                    loss_pol = torch.sum(distributions * (distributions.log() - (1 - dones[:, None]) * self.discount * (
                                1 / self.entropy_scale) * (values_next_states.detach())), 1).mean()

                    loss = self.actor_critic_factor * loss_pol + (1 - self.actor_critic_factor) * loss_value

                    # Update parameters
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    if self.use_target:
                        moving_average(self.target.parameters(), self.network.parameters(),
                                       self.target_smoothing_factor)