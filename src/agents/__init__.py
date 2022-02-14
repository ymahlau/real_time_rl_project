import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Network(nn.Module):
    """Neural network with variable dimensions"""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256, num_hidden: int = 2):
        super().__init__()

        if num_hidden < 1:
            raise ValueError("num_hidden must be greater or equal to 1 and not {num_hidden}")
        if hidden_size < 1:
            raise ValueError("hidden_size must be greater or equal to 1 and not {hidden_size}")

        self.num_hidden = num_hidden
        self.hidden_size = hidden_size

        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        additional_hidden_layers = []
        for _ in range(num_hidden - 1):
            additional_hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            additional_hidden_layers.append(nn.ReLU())
        self.additional_hidden_layers = nn.Sequential(*additional_hidden_layers)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        x = self.input(x)
        x = F.relu(x)
        if self.num_hidden > 1:
            x = self.additional_hidden_layers(x)
        x = self.output(x)
        return x
