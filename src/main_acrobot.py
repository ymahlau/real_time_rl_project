import sys

import gym

from src.main import experiment_sac, experiment_rtac


def main(seed: int = 0):
    iter_per_track = 10

    # a
    experiment_sac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, iter_per_track=iter_per_track)

    # b
    experiment_rtac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, iter_per_track=iter_per_track)

    # c
    experiment_sac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, use_rtmdp=True,
                   iter_per_track=iter_per_track)


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main(int(sys.argv[1]))
    else:
        print('No seed input found')
