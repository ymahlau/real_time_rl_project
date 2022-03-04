import sys

from src import experiment_sac, experiment_rtac
from src.envs.custom_lunar_lander import CustomLunarLander
from src.utils.wrapper import PreviousActionWrapper


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
        experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate,
                            use_device=True, flags=f'-{step_size}')
        # sac in RTMDP(E)
        experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_rtmdp=True,
                            use_device=True, flags=f'-{step_size}')
        # RTAC
        experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                             'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                             lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=True,
                             flags=f'-{step_size}')

        # sac in PreviousAction(E)
        experiment_sac(PreviousActionWrapper(CustomLunarLander(step_size=step_size), 0),
                            PreviousActionWrapper(CustomLunarLander(step_size=step_size), 0),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_rtmdp=False,
                            use_device=True, flags=f'-{step_size}-prev')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_custom_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
