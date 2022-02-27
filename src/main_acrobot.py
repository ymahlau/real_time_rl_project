import sys

import gym

from src.main import experiment_sac, experiment_rtac


def main(seed: int = 0):
    experiment_sac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed)
    experiment_rtac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed)
    experiment_sac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, use_rtmdp=True)


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main(int(sys.argv[1]))
    else:
        print('No seed input found')
