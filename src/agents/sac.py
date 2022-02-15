import gym
import torch
from torch import Tensor

from src.agents.networks import PolicyNetwork, ValueNetwork
from src.utils.utils import ReplayBuffer, evaluate_policy


class SAC:
    def __init__(
            self,
            env: gym.Env,
            alpha: float = 0.2,
            gamma: float = 0.99,
            lr_pol: float = 0.0003,
            lr_val: float = 0.003,
            replay_size: int = 10000,
            batch_size: int = 256,
            hidden_size: int = 256,
            num_hidden: int = 2):
        self.env = env

        self.alpha = alpha
        self.gamma = gamma
        self.lr_pol = lr_pol
        self.lr_val = lr_val
        self.replay_size = replay_size
        self.batch_size = batch_size

        self.action_space = self.env.action_space.n

        self.value = ValueNetwork(self.env.observation_space.shape[0] + 1,
                                  hidden_size=hidden_size,
                                  num_hidden=num_hidden)
        self.policy = PolicyNetwork(self.env.observation_space.shape[0],
                                    self.env.action_space.n,
                                    hidden_size=hidden_size,
                                    num_hidden=num_hidden)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.lr_val)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr_pol)

    def value_loss(
            self,
            states: Tensor,
            actions: Tensor,
            rewards: Tensor,
            next_states: Tensor,
            dones: Tensor) -> Tensor:
        with torch.no_grad():
            next_actions_dist = self.policy.get_action_distribution(states)
            next_actions = torch.tensor([[self.policy.act(s)] for s in states])
            next_actions_dist = torch.gather(next_actions_dist, 1, next_actions)  # keep only corresponding prob
            value_targets = self.value(torch.cat((next_states, next_actions), dim=1))

            targets = (rewards + self.gamma * (1 - dones) *
                       (value_targets - self.alpha * next_actions_dist.log())).detach()

        values = self.value(torch.cat((states, actions), dim=1))
        return torch.pow(values - targets, 2).mean()

    def policy_loss(self, states: Tensor) -> Tensor:
        next_actions_dist = self.policy.get_action_distribution(states)
        values = [self.value(torch.cat((states, (torch.ones(self.batch_size)[:, None]*a)), dim=1)) for a in range(self.action_space)]
        values = torch.squeeze(torch.stack(values, dim=1), dim=2)

        return torch.sum(next_actions_dist * (next_actions_dist.log() - (1/self.alpha) * values), 1).mean()

    def update_step(self, replay: ReplayBuffer):
        # get samples and sort into batches
        samples = replay.sample(self.batch_size)
        state_batch = torch.tensor([s[0] for s in samples])
        action_batch = torch.tensor([s[1] for s in samples])
        reward_batch = torch.tensor([s[2] for s in samples])
        next_state_batch = torch.tensor([s[3] for s in samples])
        done_batch = torch.tensor([s[4] for s in samples], dtype=torch.float)

        value_loss = self.value_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        policy_loss = self.policy_loss(state_batch)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def learn(self, iterations: int = 10000, printEval: bool = False):
        replay = ReplayBuffer(self.replay_size)

        for i in range(iterations):
            steps = 0
            done = False
            state = self.env.reset()
            while not done:
                action = self.policy.act(torch.tensor(state))
                next_state, reward, done, _ = self.env.step(action)
                replay.add_data((state, [action], [reward], next_state, [done]))
                state = next_state

                steps += 1
                if replay.capacity_reached():
                    self.update_step(replay)

                    if printEval and (steps % 100 == 0 or (done and i % 100 == 0)):
                        print(evaluate_policy(self.policy.act, self.env, rtmdp_ob=False))
