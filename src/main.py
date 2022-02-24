import copy
import sys
from typing import Union, Type

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
                    use_shared: bool = False):
    env = RTMDP(env, 0)
    eval_env = RTMDP(eval_env, 0)
    _experiment(env, eval_env, name, RTAC, steps, track_rate, seed, use_target, use_double, use_normalization,
                use_shared)


def experiment_sac(env: gym.Env,
                   eval_env: gym.Env,
                   name: str,
                   steps: int = 100000,
                   track_rate: int = 500,
                   seed: int = 0,
                   use_target: bool = False,
                   use_double: bool = False,
                   use_normalization: bool = False):
    _experiment(env, eval_env, name, SAC, steps, track_rate, seed, use_target, use_double, use_normalization, False)


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
                use_shared: bool = False):
    print(f'Start experiment {name} with seed {seed} and agent {agent.__name__}')

    suffix = ''
    network_kwargs = {}
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

    alg = agent(env, network_kwargs=network_kwargs, eval_env=eval_env, seed=seed, use_target=use_target)

    # saved with name "{agents name}-S{seed}-T{number of datapoints}-{used variation}"
    # at directory "/experiment_data/{experiment name}"
    perform_experiment(alg, f"{agent.__name__}-S{seed}-T{int(steps / track_rate)}{suffix}",
                       f"../experiment_data/{name}",
                       f"../model_data/{name}/{agent.__name__}-S{seed}-T{int(steps / track_rate)}{suffix}",
                       num_steps=steps,
                       track_rate=track_rate)

    print(f'Finished experiment {name} with seed {seed} and agent {agent.__name__}')


def main(seed: int = 0):
    experiment_rtac(gym.make('Acrobot-v1'), gym.make('Acrobot-v1'), 'Acrobot', seed=seed, use_target=True,
                    use_double=True, use_shared=False, use_normalization=True)
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


if __name__ == '__main__':
    if len(sys.argv[0]) > 1:
        main(int(sys.argv[1]))
    else:
        main()
