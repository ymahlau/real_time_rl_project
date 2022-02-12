import random
from collections import deque
import gym
import numpy as np

class ReplayBuffer:
    
    def __init__(self,capacity):
        self.replay_buffer = deque(maxlen = capacity)
        
    def add_data(self,data):
        self.replay_buffer.append(data)
    
    def capacity_reached(self):
        return len(self.replay_buffer) >= self.replay_buffer.maxlen
    
    def sample(self, sample_size):
        return random.sample(self.replay_buffer, sample_size)
   
"""
Converts the observation tuple (s,a) returned by rtmdp's
into a single sequence s + one_hot_encoding(a)
"""
def flatten_rtmdp_obs(obs,num_actions):
    #one-hot
    one_hot = np.zeros(num_actions)
    one_hot[obs[1]] = 1
    return list(obs[0]) + list(one_hot)

def evaluate_policy(policy,env, trials = 10):
    
    cum_rew = 0
    for _ in range(trials):
        state = env.reset()
        done = False
        while not done:
            state = flatten_rtmdp_obs(state,env.action_space.n)
            action = policy(state)
            state,reward,done,_ = env.step(action)
            cum_rew += reward

    return cum_rew / trials
