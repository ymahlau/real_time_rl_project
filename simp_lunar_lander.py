import gym
import numpy as np
import random
import pygame
from pygame import gfxdraw
import time

class SimpLunarLander(gym.Env):

    
    def __init__(self,step_size):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32 )
        self.reward_range = (0,100)
        self.state = None
        self.step_size = step_size
        self.engine_acceleration = 2
        self.gravity = 1
        self.playground_width = 150
        self.playground_height = 100
        self.landing_pad = [35,65]
        self.crash_threshold = 5
        
        #GUI only
        self.last_action = None
        self.frame = None
        self.screen_width = 750
        self.screen_height = 500

    def reset(self):
        self.state = [random.randint(0,self.playground_width),0,0,0]        
        return self.state

    def step(self, action):
        
        if self.state is None:
            raise AssertionError("Environment is not reset yet.")
        
        self.last_action = action
        
        #update velocities
        if action == 1:
            self.state[2] += self.engine_acceleration * self.step_size
        elif action == 2:
            self.state[2] -= self.engine_acceleration * self.step_size
        elif action == 3:
            self.state[3] -= self.engine_acceleration * self.step_size
        
        self.state[3] += self.gravity * self.step_size
        
        #update positions
        self.state[0] += self.step_size * self.state[2]
        self.state[1] += self.step_size * self.state[3]
        
        #check termination conditions
        
        #Out of bounds
        if self.state[0] < 0 or self.state[0] > self.playground_width or self.state[1] < 0:
            return self.state,-10,True,{}
        
        #Landed
        if self.state[1] >= self.playground_height:
            reward = 0
            
            if self.landing_pad[0] < self.state[0] and self.state[0] < self.landing_pad[1]:
                reward += 100
            
            if self.state[3] > self.crash_threshold:
                reward -= 20
            reward -= self.state[3] #Further increase?
            
            return self.state,reward , True, {}
        
        #Nothing happened
        return self.state,-self.step_size / 4,False,{}
        
        
    def render(self,mode):
        
        if self.frame is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((0, 0, 0))
        
        xr = self.screen_width/self.playground_width
        yr = self.screen_height/ self.playground_height
        
        #Lunar lander
        size_on_screen = self.screen_width / 40
        gfxdraw.filled_circle(
            canvas,
            int(self.state[0] * xr),
            int(self.state[1] * yr),
            int(size_on_screen),
            (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
        ) 
        
        
        #Exhaustion gases
        if self.last_action != 0:
            
            offset_x = 0
            offset_y = 0
            
            if self.last_action == 1:
                offset_x = -3/2 * size_on_screen
            elif self.last_action == 2:
                offset_x = 3/2 * size_on_screen
                
            if self.last_action == 3:
                offset_y = 3/2 *size_on_screen
            
            gfxdraw.filled_circle(
                canvas,
                int(self.state[0] * xr + offset_x),
                int(self.state[1] * yr + offset_y),
                int(size_on_screen / 4 ),
                (255, 255, 255),
            )
        
        #Landing pad
        gfxdraw.filled_circle(
            canvas,
            int(self.landing_pad[0] * xr),
            int(self.playground_height * yr),
            int(self.screen_width / 80),
            (255, 0, 0),
        ) 
        gfxdraw.filled_circle(
            canvas,
            int(self.landing_pad[1] * xr),
            int(self.playground_height * yr),
            int(self.screen_width / 80),
            (255, 0, 0),
        ) 

        self.screen.blit(canvas, (0, 0))
        pygame.display.flip()


    def close(self):
        if self.frame is not None:
            pygame.quit()
 
"""
env = SimpLunarLander(0.1)
env.reset()
done = False
while not done:
    _,_,done,_ = env.step(1)
    env.render()
    time.sleep(0.01)
"""
