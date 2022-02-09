import random
from collections import deque

class ReplayBuffer:
    
    def __init__(self,capacity):
        self.replay_buffer = deque(maxlen = capacity)
        
    def add_data(self,data):
        self.replay_buffer.append(data)
    
    def capacity_reached(self):
        return len(self.replay_buffer) >= self.replay_buffer.maxlen
    
    def sample(self, sample_size):
        return random.sample(self.replay_buffer, sample_size)

