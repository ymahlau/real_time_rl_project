import sys

from src import main
from src.envs.custom_lunar_lander import CustomLunarLander


def temp(seed: int):
    iter_per_track = 5

    step_size = 0.2
    network_kwargs = {'num_layers': 2, 'hidden_size': 512}
    seed = 51
    main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                        'CustomLunarLander', seed=seed, network_kwargs=network_kwargs, lr=0.00005,
                        entropy_scale=0.01,
                        use_target=False, use_double=False, use_normalization=False, use_rtmdp=False,
                        steps=100000, iter_per_track=iter_per_track, track_rate=5000, use_device=True)
    # main.experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
    #                      'CustomLunarLander', seed=seed, entropy_scale=0.2, network_kwargs=network_kwargs,
    #                      use_target=True, use_double=True, use_normalization=False, use_shared=False, lr=0.0005,
    #                      steps=300000, iter_per_track=iter_per_track, track_rate=5000, use_device=False)


def main_custom_lunar_lander(seed: int):
    iter_per_track = 10
    network_kwargs = {'num_layers': 2, 'hidden_size': 512}
    lr = 0.00005
    steps = 100000
    track_rate = 1000
    step_sizes = [0.01, 0.03, 0.1, 0.3, 1]

    for step_size in step_sizes:
        entropy_scale = step_size / 20

        # sac in E
        main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate,
                            use_device=True)
        # sac in RTMDP(E)
        main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_rtmdp=True,
                            use_device=True)
        # RTAC
        main.experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                             'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                             lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_custom_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
