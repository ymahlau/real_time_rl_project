from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any

import gym
import torch
from torch import Tensor

from src.agents.networks import PolicyValueNetwork
from src.utils.utils import ReplayBuffer


class ActorCritic(ABC):
    def __init__(
            self,
            env: gym.Env,
            input_size: int,
            buffer_size: int = 10000,
            use_target: bool = False,
            batch_size: int = 256,
            discount_factor: float = 0.99,
            hidden_size: int = 256,
    ):
        # environment
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")
        self.env = env
        self.input_size = input_size
        self.num_actions = env.action_space.n
        self.discount_factor = discount_factor

        # networks
        self.network = PolicyValueNetwork(input_size, self.num_actions, hidden_size=hidden_size)
        self.use_target = use_target
        if use_target:
            self.target_network = PolicyValueNetwork(input_size, self.num_actions, hidden_size=hidden_size)
        else:
            self.target_network = None

        # buffer
        self.buffer_size = buffer_size
        if buffer_size < batch_size:
            raise ValueError('Buffer has to contain more item than one batch')
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

    @abstractmethod
    def act(self, obs: Any) -> Tensor:
        pass

    @abstractmethod
    def get_value(self, obs: Any) -> Tensor:
        pass

    @abstractmethod
    def get_action_distribution(self, obs: Any) -> Tensor:
        pass

    @abstractmethod
    def update(self, samples: List[Tuple[Any, int, float, Any, bool]]):
        pass

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

    def evaluate(self,
                 logging_rate: int = 10,
                 num_episodes: int = 100,
                 checkpoint: Optional[str] = None,
                 log_dest: Optional[str] = None,
                 log_progress: bool = False,
                 ) -> float:
        return self._train_loop(
            train=False,
            logging_rate=logging_rate,
            num_episodes=num_episodes,
            checkpoint=checkpoint,
            log_dest=log_dest,
            log_progress=log_progress
        )

    def train(self,
              logging_rate: int = 10,
              num_episodes: int = 100,
              checkpoint: Optional[str] = None,
              log_dest: Optional[str] = None,
              log_progress: bool = False,
              ) -> None:
        self._train_loop(
            train=True,
            logging_rate=logging_rate,
            num_episodes=num_episodes,
            checkpoint=checkpoint,
            log_dest=log_dest,
            log_progress=log_progress
        )

    def _train_loop(
            self,
            train: bool = True,
            logging_rate: int = 10,
            num_episodes: int = 100,
            checkpoint: Optional[str] = None,
            log_dest: Optional[str] = None,
            log_progress: bool = False,
    ) -> Optional[float]:

        if checkpoint is not None:
            self.load_network(checkpoint)

        env_steps = 0

        for episode in range(1, num_episodes + 1):

            done = False
            state = self.env.reset()

            while not done:
                # logging

                # Perform step on env and add step data to replay buffer
                action = self.act(state).item()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_data((state, action, reward, next_state, done))
                state = next_state

                # update
                if train and self.buffer.capacity_reached():
                    samples = self.buffer.sample(self.batch_size)
                    self.update(samples)

        return None
