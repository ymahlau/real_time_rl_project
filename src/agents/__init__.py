from abc import ABC, abstractmethod
import time
from typing import Optional, Union, Tuple, List, Any

import gym
import numpy as np
import torch
from torch import Tensor

from src.agents.networks import PolicyValueNetwork
from src.utils.utils import flatten_rtmdp_obs, ReplayBuffer


class ActorCritic(ABC):
    def __init__(
            self,
            env: gym.Env,
            network: PolicyValueNetwork,
            buffer_size: int = 10000,
            target_network: Optional[PolicyValueNetwork] = None,
            batch_size: int = 256,
            ):
        # environment
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")
        self.env = env
        self.input_size = env.action_space.n
        self.num_actions = env.action_space.n

        # networks
        self.network = network
        self.target_network = target_network

        # buffer
        if buffer_size < batch_size:
            raise ValueError('Buffer has to contain more item than one batch')
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

    @abstractmethod
    def act(self, obs: Union[Tuple, np.ndarray]) -> Tensor:
        pass

    @abstractmethod
    def predict(self, obs: Union[Tuple, np.ndarray]) -> Tensor:
        pass

    @abstractmethod
    def update(self, samples: List[Tuple[Any, float, Any, bool]]):
        pass

    @property
    def uses_target(self):
        return self.target_network is not None

    def load_network(self, checkpoint: str):
        """
            Loads the model with parameters contained in the files in the
            path checkpoint.

            checkpoint: Absolute path without ending to the two files the model is saved in.
        """
        self.network.load_state_dict(torch.load(f"{checkpoint}.model"))
        if self.uses_target:
            self.target_network.load_state_dict(torch.load(f"{checkpoint}.model"))
        print(f"Continuing training on {checkpoint}.")

    def save_network(self, log_dest: str):
        """
           Saves the model with parameters to the files referred to by the file path log_dest.
           log_dest: Absolute path without ending to the two files the model is to be saved in.
       """
        torch.save(self.network.state_dict(), f"{log_dest}.model")
        print("Saved current training progress")

    def train(
            self,
            logging_rate: int = 10,
            num_episodes: int = 100,
            checkpoint: Optional[str] = None,
            log_dest: Optional[str] = None,
            log_progress: bool = False,
            ) -> None:

        if checkpoint is not None:
            self.load_network(checkpoint)

        env_steps = 0

        for episode in range(1, num_episodes + 1):

            done = False
            state = self.env.reset()

            while not done:
                # logging

                # Perform step on env and add step data to replay buffer
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_data((state, reward, next_state, done))
                state = next_state

                # update
                if self.buffer.capacity_reached():  # TODO: update if enough samples?
                    samples = self.buffer.sample(self.batch_size)
                    self.update(samples)








