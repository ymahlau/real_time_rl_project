import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Network(nn.Module):
    """Define policy network"""

    def __init__(self, input_size, output_size, hidden_size=128, num_hidden = 1):
        super().__init__()
        
        if num_hidden < 1:
            raise ValueError("num_hidden must be greater or equal to 1 and not {num_hidden}")
        if hidden_size < 1:
            raise ValueError("hidden_size must be greater or equal to 1 and not {hidden_size}")

        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
        
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
        return x

        
class PolicyNetwork(Network):
    
    def __init__(self, input_size, nom_actions, hidden_size = 128, num_hidden = 1):
        super().__init__(input_size,nom_actions,hidden_size = hidden_size, num_hidden = num_hidden)
        
    def get_action_distribution(self,state):
        return F.softmax(self.forward(state))
    
    def act(self,state,return_distribution = False):
        act_distr = Categorical(self.get_action_distribution(state))
        chosen_action = act_distr.sample().item()
        if return_distribution:
            return chosen_action,act_distr
        else:
            return chosen_action
        
        
