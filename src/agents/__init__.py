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
            buffer_size: int = 10000,
            use_target: bool = False,
            double_value: bool = False,
            batch_size: int = 256,
            discount_factor: float = 0.99,
            reward_scaling_factor: float = 1.0,
    ):
        # environment
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")
        self.env = env
        self.num_actions = env.action_space.n
        self.discount_factor = discount_factor
        self.reward_scaling_factor = reward_scaling_factor

        # network
        self.use_target = use_target
        self.double_value = double_value

        # buffer
        self.buffer_size = buffer_size
        if buffer_size < batch_size:
            raise ValueError('Buffer has to contain more item than one batch')
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

    @abstractmethod
    def act(self, obs: Any) -> int:
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

    @abstractmethod
    def load_network(self, checkpoint: str):
        pass

    @abstractmethod
    def save_network(self, log_dest: str):
        pass

    def evaluate(self,
                 logging_rate: int = 10,
                 num_steps: int = 100,
                 checkpoint: Optional[str] = None,
                 log_dest: Optional[str] = None,
                 log_progress: bool = False,
                 ) -> float:
        return self._train_loop(
            train=False,
            logging_rate=logging_rate,
            num_steps=num_steps,
            checkpoint=checkpoint,
            log_dest=log_dest,
            log_progress=log_progress
        )

    def train(self,
              logging_rate: int = 10,
              num_steps: int = 100,
              checkpoint: Optional[str] = None,
              log_dest: Optional[str] = None,
              log_progress: bool = False,
              ) -> None:
        self._train_loop(
            train=True,
            logging_rate=logging_rate,
            num_steps=num_steps,
            checkpoint=checkpoint,
            log_dest=log_dest,
            log_progress=log_progress
        )

    def _train_loop(
            self,
            train: bool = True,
            logging_rate: int = 10,
            num_steps: int = 100,
            checkpoint: Optional[str] = None,
            log_dest: Optional[str] = None,
            log_progress: bool = False,
    ) -> Optional[float]:

        if checkpoint is not None:
            self.load_network(checkpoint)

        cum_reward = 0
        env_steps = 0
        num_episodes = 0

        while env_steps < num_steps:

            done = False
            state = self.env.reset()

            while not done:
                # logging

                # Perform step on env and add step data to replay buffer
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_data((state, action, self.reward_scaling_factor*reward, next_state, done))

                state = next_state
                cum_reward += reward
                env_steps += 1

                # update
                if train and self.buffer.capacity_reached():
                    samples = self.buffer.sample(self.batch_size)
                    self.update(samples)

            num_episodes += 1

        if not train:
            return cum_reward / num_episodes
