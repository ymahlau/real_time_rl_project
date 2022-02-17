from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any

import gym
import torch
from torch import Tensor
from tqdm import tqdm

from src.agents.networks import PolicyValueNetwork
from src.utils.utils import ReplayBuffer

class ActorCritic(ABC):
    def __init__(
            self,
            env: gym.Env,
            eval_env: Optional[gym.Env] = None,
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
        self.eval_eval = eval_env
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
                 iterations: int = 100,
                 progress_bar:bool = False,
                 ) -> float:
        return self.env_loop(
            train=False,
            iterations=iterations,
            progress_bar=progress_bar,
        )

    def train(self,
              num_steps: int = 100,
              checkpoint: Optional[str] = None,
              save_dest: Optional[str] = None,
              save_rate: int = 10,
              track_stats: bool = False,
              track_rate: int = 100,
              progress_bar: bool=False,
              ) -> Optional[List]:
        return self.env_loop(
            train = True,
            num_steps=num_steps,
            checkpoint=checkpoint,
            save_dest=save_dest,
            save_rate=save_rate,
            track_stats = track_stats,
            track_rate = track_rate,
            progress_bar=progress_bar,
        )

    def env_loop(
            self,
            train: bool = True,
            num_steps: int = 100,
            iterations: int = 1,
            checkpoint: Optional[str] = None,
            save_dest: Optional[str] = None,
            save_rate: int = 1000, #Every how many steps the model is saved
            track_stats: bool = False,
            track_rate: int = 100, #Every how many steps the average reward is calculated
            progress_bar: bool = False,
    ) -> Optional[List]:

        if checkpoint is not None:
            self.load_network(checkpoint)

        if self.eval_eval is None:
            if track_stats:
                raise ValueError("An evaluation environment has to be specified if tracking statistics.")
            if not train:
                raise ValueError("An evaluation environment has to be specified when in evaluation mode.")

        if progress_bar:
            pbar = tqdm(total=(num_steps if train else iterations))

        env = self.env if train else self.eval_eval
        env_steps = 0
        num_episodes = 0
        performances = []
        cum_reward = 0

        while (train and env_steps < num_steps) or (not train and num_episodes < iterations):

            done = False
            state = env.reset()

            while not done:

                # Perform step on env and add step data to replay buffer
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.buffer.add_data((state, action, self.reward_scaling_factor*reward, next_state, done))

                state = next_state
                cum_reward += reward

                if train:
                    # Log current performances
                    if track_stats and env_steps % track_rate == 0:
                        avg = self.evaluate()
                        performances.append([env_steps, avg])

                    # Save current model if necessary
                    if save_dest is not None and env_steps % save_rate == 0:
                        self.save_network(save_dest)

                    # update
                    if self.buffer.capacity_reached():
                        samples = self.buffer.sample(self.batch_size)
                        self.update(samples)

                    #update progress bar
                    if progress_bar:
                        pbar.update(1)

                env_steps += 1

            num_episodes += 1
            #update progress bar
            if progress_bar and not train:
                pbar.update(1)

        if train:
            return performances if track_stats else None
        else:
            return cum_reward / iterations
