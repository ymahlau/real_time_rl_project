from vacuum import VacuumEnv
import probeEnvironments
from network import PolicyNetwork, ValueNetwork
from utilities import ReplayBuffer
import rtmdp
import numpy as np
import torch
import gym
import math
import torch.optim as optim
import random
import torch.nn as nn
from torch.distributions import Categorical
from utilities import evaluate_policy,flatten_rtmdp_obs

class RTAC:
    
    def __init__(self,env, test_env, alpha = 0.001, gamma = 0.99, lr_pol = 0.00003, lr_val = 0.0003, buffer_size = 10000, batch_size = 256, hidden_size = 256, num_hidden = 2):
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.LR_POL = lr_pol
        self.LR_VAL = lr_val
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size

        if not isinstance(env.action_space,gym.spaces.Discrete):
            raise ValueError("Action space is not discrete!")
        
        self.env = env
        self.test_env = test_env
        self.nom_actions = env.action_space.n
    
        self.pol_network = PolicyNetwork(env.observation_space[0].shape[0] + env.action_space.n,env.action_space.n, hidden_size = hidden_size, num_hidden = num_hidden)
        self.val_network = ValueNetwork(env.observation_space[0].shape[0] + env.action_space.n, hidden_size = hidden_size, num_hidden = num_hidden)
        
        self.replay = ReplayBuffer(buffer_size)

        self.optim_pol = optim.Adam(self.pol_network.parameters(), lr=lr_pol)
        self.optim_val = optim.Adam(self.val_network.parameters(), lr=lr_val)

        self.mse_loss = nn.MSELoss()

    def train(self, checkpoint = None):

        env_steps = 0
        while True:
        
            done = False
            state = self.env.reset()

            while not done:
                
                state = flatten_rtmdp_obs(state,self.nom_actions)
                action = self.pol_network.act(state)
                next_state,reward,done,_ = self.env.step(action)
                self.replay.add_data( (state,reward,next_state[0],done) )
                state = next_state
                
                env_steps += 1
                if env_steps % 100 == 0:
                    print(env_steps)
                    evaluate_policy(self.pol_network.act,self.test_env)
                
                """
                Perform single update step
                """
                if self.replay.capacity_reached():
                    sample = np.array(self.replay.sample(self.BATCH_SIZE),dtype=object)
                
                    """
                    Retrieve relevant tensors from the sample batch
                    """
                    states = []
                    for obs in sample[:,0]:
                        states.append(torch.tensor(obs))
                    states = torch.stack(states, dim=0) #tensor of states
                    
                    distributions = self.pol_network.get_action_distribution(states) #tensor of action distributions on states
        
                    rewards = torch.tensor(np.array(sample[:,1],dtype=float)) #reward tensor
                    
                    next_states_obs = []
                    for obs in sample[:,2]:
                        next_states_obs.append(torch.tensor(obs))
                    next_states_obs = torch.stack(next_states_obs, dim=0) #tensor of next states without the action part
                    
                    values_next_states = []
                    for j in range(self.nom_actions):
                        values_next_states.append(self.val_network(torch.tensor([flatten_rtmdp_obs( (list(next_states_obs[i]),j),self.nom_actions  ) for i in range(self.BATCH_SIZE)])))
                    values_next_states = torch.squeeze(torch.stack(values_next_states,dim=1),dim=2) #tensor of values of next states with all actions
                    
                    dones = torch.tensor(np.array(sample[:,3],dtype=float))
                    
                    """
                    Calculate loss functions now
                    """
                    #calc value func loss
                    targets = (rewards + torch.sum(distributions * ( (1 - dones[:,None]) * self.GAMMA * values_next_states - self.ALPHA * distributions.log() ),1)).detach().float()
                    values = torch.squeeze(self.val_network(states),1)
                    loss_value = self.mse_loss(values,targets)
                    
                    #calc policy func loss
                    loss_pol = torch.sum(distributions * (distributions.log() - (1-dones[:,None]) * self.GAMMA*(1/self.ALPHA)* (values_next_states.detach())),1).mean()
                    
                    """
                    Update parameters
                    """
                    self.optim_pol.zero_grad()
                    loss_pol.backward()
                    self.optim_pol.step()
                    
                    self.optim_val.zero_grad()
                    loss_value.backward()
                    self.optim_val.step() 


#env = rtmdp.RTMDP(probeEnvironments.ConstRewardEnv(),0)
#env = rtmdp.RTMDP(gym.make('LunarLander-v2'),0)
#env = rtmdp.RTMDP(probeEnvironments.PredictableRewardEnv(),0)
#env = rtmdp.RTMDP(probeEnvironments.TwoStatesActionsEnv(),0)
#env = rtmdp.RTMDP(probeEnvironments.TwoStateDependentActions(),0)
env = rtmdp.RTMDP(gym.make('CartPole-v1'),0)
test_env = rtmdp.RTMDP(gym.make('CartPole-v1'),0)

test = RTAC(env,test_env)
test.train()

