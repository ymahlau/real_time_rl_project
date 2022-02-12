from network import PolicyNetwork, ValueNetwork
from utilities import ReplayBuffer
import probeEnvironments
import torch
import gym

class SAC:
    def __init__(self, env, alpha=1, gamma=0.99, learning_rate=3e-4, replay_size=10000, batch_size=256):
        self.env = env

        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.replay_size = replay_size
        self.batch_size = batch_size

        self.action_space = self.env.action_space.n

        self.value = ValueNetwork(self.env.observation_space.shape[0] + 1)
        self.policy = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.learning_rate)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def value_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions_dist = self.policy.get_action_distribution(states)
            next_actions = torch.tensor([[self.policy.act(s)] for s in states])
            next_actions_dist = torch.gather(next_actions_dist, 1, next_actions)  # keep only corresponding prob
            value_targets = self.value(torch.cat((next_states, next_actions), dim=1))

            targets = (rewards + self.gamma * (1 - dones) *
                       (value_targets - self.alpha * next_actions_dist.log())).detach()

        values = self.value(torch.cat((states, actions), dim=1))
        return torch.pow(values - targets, 2).mean()

    def policy_loss(self, states):
        next_actions_dist = self.policy.get_action_distribution(states)
        values = [self.value(torch.cat((states, (torch.ones(self.batch_size)[:, None]*a)), dim=1)) for a in range(self.action_space)]
        values = torch.squeeze(torch.stack(values, dim=1), dim=2)

        return torch.sum(next_actions_dist * (next_actions_dist.log() - (1/self.alpha) * values), 1).mean()

    def update_step(self, replay):
        # get samples and sort into batches
        samples = replay.sample(self.batch_size)
        state_batch = torch.tensor([s[0] for s in samples])
        action_batch = torch.tensor([s[1] for s in samples])
        reward_batch = torch.tensor([s[2] for s in samples])
        next_state_batch = torch.tensor([s[3] for s in samples])
        done_batch = torch.tensor([s[4] for s in samples], dtype=float)

        value_loss = self.value_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        policy_loss = self.policy_loss(state_batch)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def learn(self, iterations=10000):
        replay = ReplayBuffer(self.replay_size)

        for i in range(iterations):
            steps = 0
            done = False
            state = env.reset()
            while not done:
                action = self.policy.act(torch.tensor(state))
                next_state, reward, done, _ = env.step(action)
                replay.add_data((state, [action], [reward], next_state, [done]))
                state = next_state

                steps += 1
                if replay.capacity_reached():
                    self.update_step(replay)

                    if steps % 100 == 0 or (done and i % 100 == 0):
                        print(evaluate_policy(self.policy, env))


def evaluate_policy(policy, env, iterations=10):
    total_reward = 0
    for _ in range(iterations):
        state = env.reset()
        done = False
        while not done:
            action = policy.act(torch.tensor(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward

    return total_reward / iterations


if __name__ == '__main__':
    #env = probeEnvironments.TwoStatesActionsEnv()
    env = gym.make('CartPole-v1')

    agent = SAC(env)
    agent.learn(1000000)
