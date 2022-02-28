import sys

from src import main
from src.envs.custom_lunar_lander import CustomLunarLander


def main_custom_lunar_lander(seed: int):
    iter_per_track = 10

    step_size = 0.5
    # main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
    #                      'CustomLunarLander', seed=seed,
    #                     use_target=False, use_double=False, use_normalization=False, use_rtmdp=False,
    #                     steps=1000000, iter_per_track=iter_per_track, track_rate=2000, use_device=False)
    main.experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                         'CustomLunarLander', seed=seed,
                         use_target=True, use_double=True, use_normalization=True,
                         steps=500000, iter_per_track=iter_per_track, track_rate=2000, use_device=False)


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main_custom_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
