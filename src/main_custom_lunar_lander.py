import sys

from src import main
from src.envs.custom_lunar_lander import CustomLunarLander


def main_custom_lunar_lander(seed: int):
    iter_per_track = 5

    step_size = 0.2
    network_kwargs = {'num_layers': 2, 'hidden_size': 512}
    seed = 0
    # main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
    #                     'CustomLunarLander', seed=seed, network_kwargs=network_kwargs,
    #                     use_target=True, use_double=True, use_normalization=True, use_rtmdp=False,
    #                     steps=300000, iter_per_track=iter_per_track, track_rate=5000, use_device=True)
    main.experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                         'CustomLunarLander', seed=seed, entropy_scale=0.1, network_kwargs=network_kwargs,
                         use_target=True, use_double=True, use_normalization=False, use_shared=True, lr=0.0005,
                         steps=300000, iter_per_track=iter_per_track, track_rate=5000, use_device=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_custom_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
