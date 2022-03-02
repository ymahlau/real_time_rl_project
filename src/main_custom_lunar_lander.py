import sys

from src import main
from src.envs.custom_lunar_lander import CustomLunarLander

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
                            use_device=True, flags=f'-{step_size}')
        # sac in RTMDP(E)
        main.experiment_sac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                            'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                            lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_rtmdp=True,
                            use_device=True, flags=f'-{step_size}')
        # RTAC
        main.experiment_rtac(CustomLunarLander(step_size=step_size), CustomLunarLander(step_size=step_size),
                             'CustomLunarLander', seed=seed, entropy_scale=entropy_scale, network_kwargs=network_kwargs,
                             lr=lr, steps=steps, iter_per_track=iter_per_track, track_rate=track_rate, use_device=True,
                             flags=f'-{step_size}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_custom_lunar_lander(int(sys.argv[1]))
    else:
        print('No seed input found')
