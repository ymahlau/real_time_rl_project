import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import typing
from rtmdp import RTMDP

class Network(nn.Module):
    """Neural network with variable dimensions"""

    def __init__(self, input_size, output_size, hidden_size=256, num_hidden = 2):
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
            additional_hidden_layers.append(nn.Linear(hidden_size,hidden_size))
            additional_hidden_layers.append(nn.ReLU())
        self.additional_hidden_layers = nn.Sequential(*additional_hidden_layers)

    def forward(self, x):
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        x = self.input(x)
        x = F.relu(x)
        if self.num_hidden > 1:
            x = self.additional_hidden_layers(x)
        x = self.output(x)
        return x

        
class PolicyNetwork(Network):
    
    def __init__(self, obs_size, nom_actions, hidden_size = 256, num_hidden = 2):
        super().__init__(obs_size,nom_actions,hidden_size = hidden_size, num_hidden = num_hidden)
        
    def get_action_distribution(self,state,log = False):
        if log:
            print(self.forward(state))
        return F.softmax(self.forward(state))
    
    def act(self,state):
        act_distr = self.get_action_distribution(state)
        chosen_action = Categorical(act_distr).sample().item()
        return chosen_action
        
class ValueNetwork(Network):
    
    def __init__(self, obs_size, hidden_size = 256, num_hidden = 2):
        super().__init__(obs_size,1,hidden_size = hidden_size, num_hidden = num_hidden)    
        

class PolValModule(nn.Module):
    
    def __init__(self, obs_size, nom_actions, shared_parameters, hidden_size = 256, num_hidden = 2):
        super().__init__()
        
        self.shared_parameters = shared_parameters
        
        if shared_parameters:
            self.features = Network(obs_size,hidden_size,num_hidden = num_hidden - 1, hidden_size = 256)
            self.value = nn.Linear(hidden_size,1)
            self.policy = nn.Linear(hidden_size,nom_actions)
        else:
            self.value_network = ValueNetwork(obs_size,hidden_size = hidden_size, num_hidden = num_hidden)
            self.policy_network = PolicyNetwork(obs_size,nom_actions,hidden_size = hidden_size, num_hidden = num_hidden)
        
    def forward(self,x):
        pass
    
    
    def get_action_distribution(self,state):
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            distr = F.softmax(self.policy(features))
            return distr
        else:
            return self.policy_network.get_action_distribution(state)
    
    def act(self,state):
        act_distr = self.get_action_distribution(state)
        chosen_action = Categorical(act_distr).sample().item()
        return chosen_action
            
    
    def get_value(self,state):
        if self.shared_parameters:
            features = self.features(state)
            features = F.relu(features)
            value = self.value(features)
            return value
        else:
            return self.value_network.forward(state)
