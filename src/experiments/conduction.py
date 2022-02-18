import gym
import csv

from src.agents import ActorCritic
from src.agents.rtac import RTAC
from src.envs.probe_envs import ConstRewardEnv
from src.utils.wrapper import RTMDP


def perform_experiment(algorithm: ActorCritic, setting_description: str, save_dest: str, num_steps: int = 1e6, track_rate: int = 10e3):

    performances = algorithm.train(num_steps=num_steps, track_stats=True, track_rate=track_rate, progress_bar= True)

    f = open("{0}/{1}.csv".format(save_dest, setting_description), 'a', newline ='')
    writer = csv.writer(f)
    writer.writerow(['Step', 'Average Reward']) #Header
    writer.writerows(performances) #Data
    f.close()

"""
Usage example:

env = RTMDP(gym.make('CartPole-v1'), 0)
eval_env = RTMDP(gym.make('CartPole-v1'),0)
for i in range(10):
    alg = RTAC(env, eval_env=eval_env, use_target= True)
    perform_experiment(alg, "cartpole_with_target{0}".format(i), "experiment_data/cartpole_test_data/", num_steps = 1000, track_rate = 100)
"""
    
