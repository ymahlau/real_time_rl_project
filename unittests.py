import rtmdp
import gym
import probeEnvironments
import RTAC
import unittest
import utilities
import math



class TestRTAC(unittest.TestCase):
    
    def test_value_function(self):
        TOLERANCE = 0.01
        
        env = rtmdp.RTMDP(probeEnvironments.ConstRewardEnv(),0)
        rtac = RTAC.RTAC(env, lr_val = 0.001, buffer_size = 1, batch_size = 1, hidden_size = 256, num_hidden = 2)
        rtac.train(training_time = 1)
        self.assertTrue(abs(1-rtac.val_network([0,1]).item()) < TOLERANCE)
    
        env = rtmdp.RTMDP(probeEnvironments.PredictableRewardEnv(),0)
        rtac = RTAC.RTAC(env, lr_val = 0.001, buffer_size = 1, batch_size = 1, hidden_size = 256, num_hidden = 2)
        rtac.train(training_time = 1)
        self.assertTrue(abs(1.0-rtac.val_network([1,1]).item()) < TOLERANCE)
        self.assertTrue(abs(-1.0-rtac.val_network([-1,1]).item()) < TOLERANCE)
        
    def test_policy(self):
        TOLERANCE = 0.2
        
        #Check if optimal policy can be adopted
        env = rtmdp.RTMDP(probeEnvironments.TwoActionsTwoStates(),0)
        rtac = RTAC.RTAC(env,alpha = 0.2, gamma = 1, lr_pol = 0.0003, lr_val = 0.003, buffer_size = 100, batch_size = 8, hidden_size = 256, num_hidden = 2)
        rtac.train(training_time = 5)
        avg = utilities.evaluate_policy(rtac.pol_network.act,env,trials=1000)
 
        self.assertTrue(abs(2 - avg) < TOLERANCE)
        self.assertTrue(abs (2 - rtac.val_network([0,1,0]).item()) < TOLERANCE)
        self.assertTrue(abs (2 - rtac.val_network([1,1,0]).item()) < TOLERANCE)
        self.assertTrue(abs (1 - rtac.val_network([2,0,1]).item()) < TOLERANCE)
        self.assertTrue(abs (1 - rtac.val_network([3,1,0]).item()) < TOLERANCE)
        
        #Check if random policy is adopted when entropy is valued extremely highly
        env = rtmdp.RTMDP(probeEnvironments.TwoActionsTwoStates(),0)
        rtac = RTAC.RTAC(env,alpha = 100, gamma = 1, lr_pol = 0.003, lr_val = 0.03, buffer_size = 200, batch_size = 16, hidden_size = 256, num_hidden = 2)
        rtac.train(training_time = 5)
        avg = utilities.evaluate_policy(rtac.pol_network.act,env,trials=1000)

        self.assertTrue(abs(1.5 - avg) < TOLERANCE)
        self.assertTrue(abs (2*100*math.log(2,math.e) - rtac.val_network([0,1,0]).item()) < 10)
        self.assertTrue(abs (1*100*math.log(2,math.e) - rtac.val_network([3,1,0]).item()) < 10)
        
        
        
    
unittest.main()
