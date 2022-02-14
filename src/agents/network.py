import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from src.agents import Network


class PolicyNetwork(Network):

    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 256, num_hidden: int = 2):
        super().__init__(obs_size, num_actions, hidden_size=hidden_size, num_hidden=num_hidden)

    def get_action_distribution(self, state: Tensor, log: bool = False) -> Tensor:
        if log:
            print(self.forward(state))
        return F.softmax(self.forward(state))

    def act(self, state: Tensor) -> Tensor:
        act_dist = self.get_action_distribution(state)
        chosen_action = Categorical(act_dist).sample().item()
        return chosen_action


class ValueNetwork(Network):

    def __init__(self, obs_size: int, hidden_size: int = 256, num_hidden: int = 2):
        super().__init__(obs_size, 1, hidden_size=hidden_size, num_hidden=num_hidden)


class PolicyValueNetwork(nn.Module):

    def __init__(
            self,
            obs_size: int,
            num_actions: int,
            shared_parameters: bool,
            hidden_size: int = 256,
            num_hidden: int = 2):
        super().__init__()

        self.shared_parameters = shared_parameters

        if shared_parameters:
            self.features = Network(obs_size, hidden_size, num_hidden=num_hidden - 1, hidden_size=256)
            self.value = nn.Linear(hidden_size, 1)
            self.policy = nn.Linear(hidden_size, num_actions)
        else:
            self.value_network = ValueNetwork(obs_size, hidden_size=hidden_size, num_hidden=num_hidden)
            self.policy_network = PolicyNetwork(obs_size, num_actions, hidden_size=hidden_size, num_hidden=num_hidden)

    def forward(self, x: Tensor) -> Tensor:
        pass

    def get_action_distribution(self, state: Tensor) -> Tensor:
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            dist = F.softmax(self.policy(features))
            return dist
        else:
            return self.policy_network.get_action_distribution(state)

    def act(self, state: Tensor) -> Tensor:
        act_dist = self.get_action_distribution(state)
        chosen_action = Categorical(act_dist).sample().item()
        return chosen_action

    def get_value(self, state: Tensor) -> Tensor:
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            value = self.value(features)
            return value
        else:
            return self.value_network.forward(state)
