from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any

import gym
import torch
from torch import Tensor, optim, nn
from tqdm import tqdm

from src.agents.buffer import ReplayBuffer
from src.agents.networks import PolicyValueNetwork


class ActorCritic(ABC):
    def __init__(
            self,
            env: gym.Env,
            num_obs: int,
            network_kwargs: dict[str: Any],
            eval_env: Optional[gym.Env] = None,
            buffer_size: int = 10000,
            use_target: bool = False,
            batch_size: int = 256,
            discount_factor: float = 0.99,
            reward_scaling_factor: float = 1.0,
            lr: float = 0.0003,
    ):
        # environment
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")
        self.env = env
        self.eval_eval = eval_env
        self.num_actions = env.action_space.n
        self.num_obs = num_obs
        self.discount_factor = discount_factor
        self.reward_scaling_factor = reward_scaling_factor

        # network
        self.use_target = use_target

        self.network = PolicyValueNetwork(**network_kwargs)
        if use_target:
            self.target = PolicyValueNetwork(**network_kwargs)

        # buffer
        self.buffer_size = buffer_size
        if buffer_size < batch_size:
            raise ValueError('Buffer has to contain more item than one batch')
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.network.policy_input_size, capacity=buffer_size)

        # optimizer and loss
        self.optim = optim.Adam(self.network.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

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
    def update(self, samples: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pass

    @abstractmethod
    def obs_to_tensor(self, obs: Any) -> Tensor:
        pass

    def load_network(self, checkpoint: str):
        """
            Loads the model with parameters contained in the files in the
            path checkpoint.

            checkpoint: Absolute path without ending to the two files the model is saved in.
        """
        self.network.load_state_dict(torch.load(f"{checkpoint}.model"))
        if self.use_target:
            self.target.load_state_dict(torch.load(f"{checkpoint}.model"))
        print(f"Continuing training on {checkpoint}.")

    def save_network(self, log_dest: str):
        """
           Saves the model with parameters to the files referred to by the file path log_dest.
           log_dest: Absolute path without ending to the two files the model is to be saved in.
       """
        torch.save(self.network.state_dict(), f"{log_dest}.model")
        print("Saved current training progress")

    def evaluate(self,
                 iterations: int = 100,
                 progress_bar: bool = False,
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
              progress_bar: bool = False,
              ) -> Optional[List]:
        return self.env_loop(
            train=True,
            num_steps=num_steps,
            checkpoint=checkpoint,
            save_dest=save_dest,
            save_rate=save_rate,
            track_stats=track_stats,
            track_rate=track_rate,
            progress_bar=progress_bar,
        )

    def env_loop(
            self,
            train: bool = True,
            num_steps: int = 100,
            iterations: int = 1,
            checkpoint: Optional[str] = None,
            save_dest: Optional[str] = None,
            save_rate: int = 1000,  # Every how many steps the model is saved
            track_stats: bool = False,
            track_rate: int = 100,  # Every how many steps the average reward is calculated
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

        def training_ongoing():
            return (train and env_steps < num_steps) or (not train and num_episodes < iterations)

        env = self.env if train else self.eval_eval
        env_steps = 0
        num_episodes = 0
        performances = []
        cum_reward = 0

        while training_ongoing():

            done = False
            state = env.reset()

            while not done:

                # Perform step on env and add step data to replay buffer
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                scaled_reward = self.reward_scaling_factor * reward
                state_tensor = self.obs_to_tensor(state)
                next_state_tensor = self.obs_to_tensor(next_state)

                self.buffer.add_data((state_tensor, action, scaled_reward, next_state_tensor, done))

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

                    # update progress bar
                    if progress_bar:
                        pbar.update(1)

                env_steps += 1
                if not training_ongoing():
                    break

            num_episodes += 1
            # update progress bar
            if progress_bar and not train:
                pbar.update(1)

        if train:
            return performances if track_stats else None
        else:
            return cum_reward / iterations
