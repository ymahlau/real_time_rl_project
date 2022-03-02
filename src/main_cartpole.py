import sys

import gym

from src.main import experiment_sac, experiment_rtac


def main(seed: int = 0):
    iter_per_track = 10

    # a
    experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                   iter_per_track=iter_per_track)

    # b
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track)

    # c
    experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                   iter_per_track=iter_per_track, use_rtmdp=True)

    # d
    experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                   iter_per_track=iter_per_track, use_rtmdp=True, use_normalization=True)

    # e
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track, use_normalization=True)

    # f
    experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                   iter_per_track=iter_per_track, use_rtmdp=True, use_double=True)

    # g
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track, use_double=True)

    # h
    experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                   iter_per_track=iter_per_track, use_rtmdp=True, use_target=True)

    # i
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track, use_target=True)

    # j
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track, use_shared=True)

    # k
    experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed,
                    iter_per_track=iter_per_track, use_shared=True, use_normalization=True)


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main(int(sys.argv[1]))
    else:
        print('No seed input found')