import csv
from pathlib import Path
from typing import Union, Optional

from src.agents import ActorCritic


def perform_experiment(
        algorithm: ActorCritic,
        file_name: str,
        file_path: Union[str, Path],
        model_dest: Optional[str] = None,
        num_steps: int = 1e6,  # total number of steps
        track_rate: int = 100,  # how many steps until data point is collected
        iter_per_track: int = 100,  # how many iterations per data point collection
):
    """
    Usage example:

    env = RTMDP(gym.make('CartPole-v1'), 0)
    eval_env = RTMDP(gym.make('CartPole-v1'),0)
    for i in range(10):
        alg = RTAC(env, eval_env=eval_env, use_target= True)
        perform_experiment(alg, "cartpole_with_target{i}",
            "experiment_data/cartpole_test_data",
            num_steps = 1000, track_rate = 100)
    """
    performances = algorithm.train(num_steps=num_steps,
                                   track_stats=True,
                                   track_rate=track_rate,
                                   progress_bar=True,
                                   save_dest=model_dest,
                                   save_rate=num_steps-1,
                                   iter_per_track=iter_per_track)

    f = open(f"{file_path}/{file_name}.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(['Step', 'Average Reward'])  # Header
    writer.writerows(performances)  # Data
    f.close()
