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

        if output_size < 1:
            raise ValueError(f"output_size must be greater or equal to 1 and not {output_size}")

        if input_size < 1:
            raise ValueError(f"input_size must be greater or equal to 1 and not {input_size}")

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # special case: single layer
        if num_layers == 1:
            self.single_layer = nn.Linear(input_size, output_size)
            return

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


class PolicyValueNetwork(nn.Module):

    def __init__(
            self,
            value_input_size: int,
            policy_input_size: int,
            output_size: int,
            shared_parameters: bool = False,
            double_value: bool = False,
            normalized: bool = False,
            hidden_size: int = 256,
            num_layers: int = 2,
            pop_art_factor: float = 0.0003,
            epsilon: float = 1e-6  # for numerical stability, not given in rtac-paper
    ):
        super().__init__()

        if pop_art_factor < 0 or pop_art_factor > 1:
            raise ValueError(f'pop_art_factor factor has to be in [0, 1], but got: {pop_art_factor}')

        if shared_parameters and num_layers < 2:
            raise ValueError('With shared parameters there have to be at least two layers')

        if shared_parameters and value_input_size != policy_input_size:
            raise ValueError('With shared parameters input sizes have to be equal')

        # flags and scalars
        self.shared_parameters = shared_parameters
        self.double_value = double_value
        self.normalized = normalized
        self.pop_art_factor = pop_art_factor
        self.epsilon: Tensor = torch.tensor(epsilon)

        # network attributes
        self.value_input_size = value_input_size
        self.policy_input_size = policy_input_size
        self.output_size = output_size

        # networks
        if shared_parameters:
            self.features = Network(value_input_size, hidden_size, num_layers=num_layers - 1, hidden_size=256)
            self.value = nn.Linear(hidden_size, 1)
            if double_value:
                self.value2 = nn.Linear(hidden_size, 1)
            self.policy = nn.Linear(hidden_size, output_size)
        else:
            self.value = Network(input_size=value_input_size,
                                 output_size=1,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers)
            self.policy = Network(input_size=policy_input_size,
                                  output_size=output_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers)
            if double_value:
                self.value2 = Network(input_size=value_input_size,
                                      output_size=1,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers)

        # normalization parameter (updatable)
        self.norm_layer = nn.Linear(1, 1)
        self.norm_layer.weight.data = torch.tensor([[1]]).float()
        self.norm_layer.bias.data = torch.tensor([0]).float()
        self.scale: nn.Parameter = nn.Parameter(torch.tensor(1).float(), requires_grad=False)  # mu (mean)
        self.shift: nn.Parameter = nn.Parameter(torch.tensor(0).float(), requires_grad=False)  # sigma (std)
        self.second_moment: nn.Parameter = nn.Parameter(torch.tensor(1).float(), requires_grad=False)  # nu

    def get_action_distribution(self, state: Tensor) -> Tensor:
        if self.shared_parameters:
            feature_vals = self.features(state)
            feature_vals = F.relu(feature_vals)
        else:
            feature_vals = state

        action_scores = self.policy(feature_vals)
        if len(action_scores.shape) == 1:
            return F.softmax(action_scores, dim=0)
        elif len(action_scores.shape) == 2:
            return F.softmax(action_scores, dim=1)
        else:
            raise ValueError('Softmax of tensors with three or more dims not supported')

    def act(self, state: Tensor) -> int:
        act_dist = self.get_action_distribution(state)
        chosen_action = Categorical(act_dist).sample().item()
        return chosen_action

    def get_value(self, state: Tensor) -> Tensor:
        # feature values
        if self.shared_parameters:
            feature_vals = self.features(state)
            feature_vals = F.relu(feature_vals)
        else:
            feature_vals = state

        # get value
        value_result = self.value(feature_vals)

        # double value minimum
        if self.double_value:
            value2 = self.value2(feature_vals)
            value_result = torch.minimum(value_result, value2)

        # normalization
        if self.normalized:
            value_result = self.norm_layer(value_result)

        return value_result

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
        old_shift = self.shift.data
        old_scale = self.scale.data

        # first update normalization parameters (exponentially moving avg)
        mean_estimate = torch.mean(new_values)
        square_estimate = torch.mean(torch.pow(new_values, 2))
        self.second_moment.data = (1 - self.pop_art_factor) * self.second_moment + self.pop_art_factor * square_estimate
        self.shift.data = (1 - self.pop_art_factor) * self.shift + self.pop_art_factor * mean_estimate
        diff = torch.maximum(self.epsilon, (self.second_moment - self.shift * self.shift))
        self.scale.data = torch.sqrt(diff)

        # then update weights of norm layer
        self.norm_layer.weight.data = (old_scale / self.scale) * self.norm_layer.weight
        self.norm_layer.bias.data = (self.scale * self.norm_layer.bias + old_shift - self.shift) / self.scale
