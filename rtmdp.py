import gym
from vacuum import VacuumEnv
import numpy as np

"""
A wrapper to transform any environment into a real-time environment.
The wrapped environment is equivalent to a 1-step constant delay of the original environment.
"""
class RTMDP(gym.Wrapper):
    
    """
        env: The environment to be wrapped into a real-time environment
        initial_action: The first action that is automatically taken in this environment 
    """
    def __init__(self, env, initial_action):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple( (env.observation_space,env.action_space) )
        self.initial_action = initial_action
        self.last_action = initial_action
    
    def reset(self):
        self.last_action = self.initial_action
        s0 = super().reset()
        return (s0,self.initial_action)
    
    def step(self,action):
        s,r,d,m = super().step(self.last_action)
        self.last_action = action
        return ((s,action),r,d,m)

