import sys

import gym

from src import experiment_sac, experiment_rtac


def main_lunar_lander(seed: int):
    steps = 300000
    iter_per_track = 10
    track_rate = 500
    # # a Vanilla SAC
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=False, use_double=False, use_normalization=False, use_rtmdp=False,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    # # b Vanilla RTAC
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=False, use_double=False, use_normalization=False, use_shared=False,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # c vanilla sac in rtmdp
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=False, use_double=False, use_normalization=False, use_rtmdp=True,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # d SAC in RTMDP with Output Normalization
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=False, use_double=False, use_normalization=True, use_rtmdp=True,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # e RTAC with Output Normalization
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=False, use_double=False, use_normalization=True, use_shared=False,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # f SAC in RTMDP with Double Value
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=False, use_double=True, use_normalization=False, use_rtmdp=True,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # g RTAC with Double Value
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=False, use_double=True, use_normalization=False, use_shared=False,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # h SAC in RTMDP with Target Network
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=True, use_double=False, use_normalization=False, use_rtmdp=True,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # i RTAC with Target Network
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=True, use_double=False, use_normalization=False, use_shared=False,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # j RTAC with shared Parameters
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=False, use_double=False, use_normalization=False, use_shared=True,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # k RTAC with everything
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=True, use_double=True, use_normalization=True, use_shared=True,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)
    #
    # # l SAC with everything
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=True, use_double=True, use_normalization=True, use_rtmdp=False,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=False)

    # m SAC in RTMDP with everything
    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                        use_target=True, use_double=True, use_normalization=True, use_rtmdp=True,
                        steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=True)

    # n RTAC shared + Normalization
    experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed,
                         use_target=False, use_double=False, use_normalization=True, use_shared=True,
                         steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=True)


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
