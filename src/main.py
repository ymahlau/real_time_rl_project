import argparse
import time

import gym

from src import experiment_sac, experiment_rtac
from src.envs.custom_lunar_lander import CustomLunarLander


def main():
    parser = argparse.ArgumentParser(description='Evaluate either SAC or RTAC on an environment of choice.')
    parser.add_argument('agent', type=str, choices=['sac', 'rtac'],
                        help='the agent you want to evaluate')
    parser.add_argument('--env', default='Acrobot-v1', type=str,
                        choices=['Acrobot-v1', 'Cartpole-v1', 'LunarLander-v2', 'CustomLunarLander'],
                        help='the environment for training and evaluation')
    # parser.add_argument('--eval_env', type=str, choices=['Acrobot-v1', 'Cartpole-v1',
    #                                                      'LunarLander-v2', 'CustomLunarLander'])
    parser.add_argument('--name', type=str,
                        help='name used to save the evaluation. Affixes for extensions, seed and number '
                             'of tracks will be added automatically')
    parser.add_argument('--steps', default=100000, type=int,
                        help='total amount of steps that will be performed')
    parser.add_argument('--track_rate', '-tr', default=500, type=int,
                        help='number of steps between evaluations')
    parser.add_argument('--iter_per_track', '-ipt', default=10, type=int,
                        help='number of episodes conducted for evaluation')
    parser.add_argument('--seed', '-s', type=int,
                        help='seed to initialise random number generation.')
    parser.add_argument('--target', default=False, action='store_true',
                        help='activates target network extension')
    parser.add_argument('--double', default=False, action='store_true',
                        help='activates double network extension')
    parser.add_argument('--norm', default=False, action='store_true',
                        help='activates normalisation')
    parser.add_argument('--rtmdp', default=False, action='store_true',
                        help='activates real-time wrapper. Always active for RTAC')
    parser.add_argument('--shared', default=False, action='store_true',
                        help='activates shared network extension. Only for RTAC')
    parser.add_argument('--device', default=False, action='store_true',
                        help='activates CUDA if possible')
    parser.add_argument('--lr', default=0.0003, type=float,
                        help='learing rate of the agent')
    parser.add_argument('--entropy', default=0.2, type=float,
                        help='entropy scale of the agent')
    parser.add_argument('--step_size', default=0.2, type=float,
                        help='step size used for CustomLunarLander environment')
    args = vars(parser.parse_args())

    if args["env"] == 'CustomLunarLander':
        env = CustomLunarLander(step_size=args["step_size"])
        flags = f'-{args["step_size"]}'
    else:
        env = gym.make(args["env"])
        flags = ''
    eval_env = gym.make(args["eval_env"]) if "eval_env" in args else gym.make(args["env"])
    name = args["name"] if "name" in args else env
    seed = args["seed"] if "seed" in args else time.time_ns()

    if args["agent"] == 'sac':
        experiment_sac(env=env,
                       eval_env=eval_env,
                       name=name,
                       steps=args["steps"],
                       track_rate=args["track_rate"],
                       seed=seed,
                       use_target=args["target"],
                       use_double=args["double"],
                       use_normalization=args["norm"],
                       use_rtmdp=args["rtmdp"],
                       iter_per_track=args["iter_per_track"],
                       use_device=args["device"],
                       lr=args["lr"],
                       entropy_scale=args["entropy"],
                       flags=flags
                       )
    elif args["agent"] == 'rtac':
        experiment_rtac(env=env,
                        eval_env=eval_env,
                        name=name,
                        steps=args["steps"],
                        track_rate=args["track_rate"],
                        seed=seed,
                        use_target=args["target"],
                        use_double=args["double"],
                        use_normalization=args["norm"],
                        use_shared=args["shared"],
                        iter_per_track=args["iter_per_track"],
                        use_device=args["device"],
                        lr=args["lr"],
                        entropy_scale=args["entropy"],
                        flags=flags
                        )


if __name__ == '__main__':
    main()
