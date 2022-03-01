import sys
from typing import Union, Type, Dict, Optional

import gym

from src.agents.rtac import RTAC
from src.agents.sac import SAC
from src.experiments.logging import perform_experiment
from src.utils.wrapper import RTMDP


def experiment_rtac(env: gym.Env,
                    eval_env: gym.Env,
                    name: str,
                    steps: int = 100000,
                    track_rate: int = 500,
                    seed: int = 0,
                    use_target: bool = False,
                    use_double: bool = False,
                    use_normalization: bool = False,
                    use_shared: bool = False,
                    iter_per_track: int = 100,
                    use_device: bool = True,
                    network_kwargs: Optional[Dict] = None,
                    lr: float = 0.0003,
                    entropy_scale: float = 0.2,
                    ):
    _experiment(
        env=env,
        eval_env=eval_env,
        name=name,
        agent=RTAC,
        steps=steps,
        track_rate=track_rate,
        seed=seed,
        use_target=use_target,
        use_double=use_double,
        use_normalization=use_normalization,
        use_shared=use_shared,
        use_rtmdp=True,
        iter_per_track=iter_per_track,
        use_device=use_device,
        network_kwargs=network_kwargs,
        lr=lr,
        entropy_scale=entropy_scale,
    )


def experiment_sac(env: gym.Env,
                   eval_env: gym.Env,
                   name: str,
                   steps: int = 100000,
                   track_rate: int = 500,
                   seed: int = 0,
                   use_target: bool = False,
                   use_double: bool = False,
                   use_normalization: bool = False,
                   use_rtmdp: bool = False,
                   iter_per_track: int = 100,
                   use_device: bool = True,
                   network_kwargs: Optional[Dict] = None,
                   lr: float = 0.0003,
                   entropy_scale: float = 0.2,
                   ):
    _experiment(
        env=env,
        eval_env=eval_env,
        name=name,
        agent=SAC,
        steps=steps,
        track_rate=track_rate,
        seed=seed,
        use_target=use_target,
        use_double=use_double,
        use_normalization=use_normalization,
        use_shared=False,
        use_rtmdp=use_rtmdp,
        iter_per_track=iter_per_track,
        use_device=use_device,
        network_kwargs=network_kwargs,
        lr=lr,
        entropy_scale=entropy_scale,
    )


def _experiment(env: Union[gym.Env, RTMDP],
                eval_env: Union[gym.Env, RTMDP],
                name: str,
                agent: Union[Type[SAC], Type[RTAC]],
                steps: int = 100000,
                track_rate: int = 500,
                seed: int = 0,
                use_target: bool = False,
                use_double: bool = False,
                use_normalization: bool = False,
                use_shared: bool = False,
                use_rtmdp: bool = False,
                iter_per_track: int = 100,
                use_device: bool = True,
                network_kwargs: Optional[Dict] = None,
                lr: float = 0.0003,
                entropy_scale: float = 0.2,
                ):
    if network_kwargs is None:
        network_kwargs = {}

    print(f'Start experiment {name} with seed {seed} and agent {agent.__name__}')

    suffix = ''
    if use_target:
        suffix += '-target'
    if use_double:
        suffix += '-double'
        network_kwargs['double_value'] = True
    if use_normalization:
        suffix += '-norm'
        network_kwargs['normalized'] = True
    if use_shared:
        suffix += '-shared'
        network_kwargs['shared_parameters'] = True
    if use_rtmdp:
        suffix += '-rtmdp'
        env = RTMDP(env, 0)
        eval_env = RTMDP(eval_env, 0)

    alg = agent(env, network_kwargs=network_kwargs, eval_env=eval_env, seed=seed, use_target=use_target,
                use_device=use_device, lr=lr, entropy_scale=entropy_scale)

    # saved with name "{agents name}-S{seed}-T{number of data points}-{used variation}"
    # at directory "/experiment_data/{experiment name}"
    perform_experiment(
        algorithm=alg,
        file_name=f"{agent.__name__}-S{seed}-T{int(steps / track_rate)}{suffix}",
        file_path=f"../experiment_data/{name}",
        model_dest=f"../model_data/{name}/{agent.__name__}-S{seed}-T{int(steps / track_rate)}{suffix}",
        num_steps=steps,
        track_rate=track_rate,
        iter_per_track=iter_per_track
    )

    print(f'Finished experiment {name} with seed {seed} and agent {agent.__name__}')


def main(seed: int = 0):
    # experiment_rtac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, use_target=True,
    #                 use_double=True, use_shared=False, use_normalization=True)
    # experiment_sac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, use_target=True,
    #                use_double=True, use_normalization=True)
    #
    # experiment_rtac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed, use_target=True,
    #                 use_double=True, use_shared=False, use_normalization=True)
    # experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed, use_target=True,
    #                use_double=True, use_normalization=True)
    #
    # experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed, use_target=True,
    #                 use_double=True, use_shared=False, use_normalization=True)
    # experiment_sac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed, use_target=True,
    #                use_double=True, use_normalization=True)

    # experiment_rtac(gym.make('CartPole-v1'), gym.make('CartPole-v1'), 'CartPole', seed=seed, use_target=False,
    #                 use_double=False, use_shared=True, use_normalization=False, iter_per_track=20)

    experiment_sac(gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), 'LunarLander', seed=seed, use_target=False,
                   use_double=False, use_normalization=False, steps=200000, iter_per_track=1, track_rate=2000)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
