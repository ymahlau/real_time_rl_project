import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class Network(nn.Module):
    """Neural network with variable dimensions"""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_hidden must be greater or equal to 1 and not {num_layers}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be greater or equal to 1 and not {hidden_size}")

        # special case: single layer
        if num_layers == 1:
            self.single_layer = nn.Linear(input_size, output_size)
            return

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        hidden_layer_list = []
        for _ in range(num_layers - 2):
            hidden_layer_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_layer_list.append(nn.ReLU())
        self.hidden_layers = nn.ModuleList(hidden_layer_list)

    def forward(self, x: Tensor) -> Tensor:
        # convert input if necessary
        if not isinstance(x, Tensor):
            x = torch.tensor(x)
        x = x.float()

        # special case: single layer
        if self.num_layers == 1:
            return self.single_layer(x)

        x = self.input(x)
        x = F.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output(x)
        return x


class PolicyNetwork(Network):

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__(input_size, output_size, hidden_size=hidden_size, num_layers=num_layers)

    def get_action_distribution(self, state: Tensor, log: bool = False) -> Tensor:
        if log:
            print(self.forward(state))
        values = self.forward(state)
        if len(values.shape) == 1:
            return F.softmax(values, dim=0)
        elif len(values.shape) == 2:
            return F.softmax(values, dim=1)
        else:
            raise ValueError('Softmax of 3d tensor not supported')

    def act(self, state: Tensor) -> int:
        act_dist = self.get_action_distribution(state)
        chosen_action = Categorical(act_dist).sample().item()
        return chosen_action


class ValueNetwork(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 256,
            num_layers: int = 2,
            double_value: bool = False,
            normalized: bool = False,
            pop_art_factor: float = 0.0003,
            epsilon: float = 1e-6  # for numerical stability, not given in rtac-paper
    ):
        super().__init__()

        if pop_art_factor < 0 or pop_art_factor > 1:
            raise ValueError(f'pop_art_factor factor has to be in [0, 1], but got: {pop_art_factor}')

        # flags and scalars
        self.double_value = double_value
        self.normalized = normalized
        self.pop_art_factor = pop_art_factor
        self.epsilon = epsilon

        # networks
        self.value = Network(input_size=input_size,
                             output_size=1,
                             hidden_size=hidden_size,
                             num_layers=num_layers)
        if double_value:
            self.value2 = Network(input_size=input_size,
                                  output_size=1,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers)

        # normalization parameter (updatable)
        self.norm_layer = nn.Linear(1, 1)
        self.norm_layer.weight.data = torch.tensor([[1]]).float()
        self.norm_layer.bias.data = torch.tensor([0]).float()
        self.scale: float = 1  # mu (mean)
        self.shift: float = 0  # sigma (std)
        self.second_moment: float = 1  # nu

    def forward(self, x: Tensor) -> Tensor:
        if self.double_value:
            value = torch.minimum(self.value(x), self.value2(x))
        else:
            value = self.value(x)

        if self.normalized:
            value = self.norm_layer(value)

        return value

    def unnormalize(self, value: Tensor) -> Tensor:
        if not self.normalized:
            raise AttributeError('Cannot unnormalize if network is not normalized')
        new_value = value * self.scale + self.shift
        return new_value

    def normalize(self, value: Tensor) -> Tensor:
        if not self.normalized:
            raise AttributeError('Cannot normalize if network is not normalized')
        new_value = (value - self.shift) / self.scale
        return new_value

    @torch.no_grad()
    def update_normalization(self, new_values: Tensor):
        if not self.normalized:
            raise AttributeError('Cannot update normalization if normalization is False')

        # save old values
        old_shift = self.shift
        old_scale = self.scale
        old_second_moment = self.second_moment

        # first update normalization parameters (exponentially moving avg)
        mean_estimate = torch.mean(new_values).item()
        square_estimate = torch.mean(torch.pow(new_values, 2)).item()
        self.second_moment = (1 - self.pop_art_factor) * self.second_moment + self.pop_art_factor * square_estimate
        self.shift = (1 - self.pop_art_factor) * self.shift + self.pop_art_factor * mean_estimate
        diff = np.maximum(self.epsilon, (self.second_moment - np.power(self.shift, 2)))
        self.scale = np.sqrt(diff).item()

        # then update weights of norm layer
        self.norm_layer.weight.data = (old_scale / self.scale) * self.norm_layer.weight
        self.norm_layer.bias.data = (self.scale * self.norm_layer.bias + old_shift - self.shift) / self.scale


class PolicyValueNetwork(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            shared_parameters: bool = False,
            double_value: bool = False,
            hidden_size: int = 256,
            num_layers: int = 2):
        super().__init__()

        if shared_parameters and num_layers < 2:
            raise ValueError('With shared parameters there have to be at least two layers')

        self.shared_parameters = shared_parameters
        self.double_value = double_value
        self.input_size = input_size
        self.output_size = output_size

        if shared_parameters:
            self.features = Network(input_size, hidden_size, num_layers=num_layers - 1, hidden_size=256)
            self.value = nn.Linear(hidden_size, 1)
            if double_value:
                self.value2 = nn.Linear(hidden_size, 1)
            self.policy = nn.Linear(hidden_size, output_size)
        else:
            self.value_network = ValueNetwork(input_size=input_size,
                                              hidden_size=hidden_size,
                                              num_layers=num_layers,
                                              double_value=double_value)
            self.policy_network = PolicyNetwork(input_size=input_size,
                                                output_size=output_size,
                                                hidden_size=hidden_size,
                                                num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        pass

    def get_action_distribution(self, state: Tensor) -> Tensor:
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            prob_values = self.policy(features)
            if len(prob_values.shape) == 1:
                return F.softmax(prob_values, dim=0)
            elif len(prob_values.shape) == 2:
                return F.softmax(prob_values, dim=1)
            else:
                raise ValueError('Softmax of 3d tensor not supported')
        else:
            return self.policy_network.get_action_distribution(state)

    def act(self, state: Tensor) -> int:
        act_dist = self.get_action_distribution(state)
        chosen_action = Categorical(act_dist).sample().item()
        return chosen_action

    def get_value(self, state: Tensor) -> Tensor:
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            value = self.value(features)
            if self.double_value:
                value2 = self.value2(features)
                return torch.minimum(value, value2)
            else:
                return value
        else:
            return self.value_network(state)
