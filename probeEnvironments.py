import gym
import numpy as np

"""
Environment with one action, one observation, one timestep and one reward (+1). 
"""
class ConstRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([0]), dtype=int)

    def reset(self):
        return [0]

    def step(self, action):
        return [0], 1, True, {}

    def close(self):
        return True


"""
Environment with one action, two observation (+1/-1), one timestep and observation-dependent reward (+1/-1). 
"""
class PredictableRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=int)

        self.num = np.random.choice([-1, 1])

    def reset(self):
        self.num = np.random.choice([-1, 1])
        return [self.num]

    def step(self, action):
        return [self.num], self.num, True, {}

    def close(self):
        return True


"""
Environment with one action, two observations (0, then 1), two timesteps and reward (+1) at the end. 
"""
class TwoStepsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=int)

        self.time = 0

    def reset(self):
        self.time = 0
        return [0]

    def step(self, action):
        if self.time == 0:
            self.time += 1
            return [1], 0, False, {}
        else:
            return [0], 1, True, {}

    def close(self):
        return True


"""
Environment with two actions (0/1), observation, one timestep and action-dependent reward (+1/-1). 
"""
class TwoActionsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([0]), dtype=int)

    def reset(self):
        return [0]

    def step(self, action):
        if action == 0:
            return [0], -1, True, {}
        else:
            return [0], 1, True, {}

    def close(self):
        return True


"""
Environment with two actions (0/1), two observation (-1/+1), one timestep and (action & observation)-dependent reward (+1/-1). 
"""
class TwoStatesActionsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=int)

        self.state = np.random.choice([-1, 1])

    def reset(self):
        self.state = np.random.choice([-1, 1])
        return [self.state]

    def step(self, action):
        if action == 0 and self.state == -1:
            return [0], -1, True, {}
        elif action == 1 and self.state == -1:
            return [1], 1, True, {}
        elif action == 0 and self.state == 1:
            return [1], 1, True, {}
        elif action == 1 and self.state == 1:
            return [0], -1, True, {}

    def close(self):
        return True
