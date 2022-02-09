import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Define policy network"""

    def __init__(self, observation_space, action_space, hidden_size=128, num_hidden = 1):
        super().__init__()
        
        if num_hidden < 1:
            raise ValueError("num_hidden must be greater or equal to 1 and not {num_hidden}")
        if hidden_size < 1:
            raise ValueError("hidden_size must be greater or equal to 1 and not {hidden_size}")
        if not isinstance(action_space,spaces.Discrete):
            raise ValueError("Action space needs to be discrete")

        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        
        self.input = nn.Linear(np.asscalar(np.prod(observation_space.shape)), hidden_size)
        self.output = nn.Linear(hidden_size,action_space.n)
        
        additional_hidden_layers = []
        for _ in range(num_hidden -1):      
            additional_hidden_layers.append(nn.Linear(hidden_dim,hidden_dim))
            additional_hidden_layers.append(nn.ReLU())
        self.additional_hidden_layers = nn.Sequential(*additional_hidden_layers)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.input(x)
        x = F.relu(x)
        if self.num_hidden > 1:
            x = self.additional_hidden_layers(x)
        x = self.output(x)
        x = F.softmax(x)
        return x
    
    def act(self,state):
        act_distr = Categorical(self.forward(state))
        chosen_action = act_distr.sample().item()
        return chosen_action
        
