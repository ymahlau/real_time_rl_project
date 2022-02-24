import os
import unittest
from pathlib import Path

from src.agents.sac import SAC
from src.envs.probe_envs import TwoActionsTwoStates
from src.experiments.analysis import analyse_experiments, visualize_statistics
from src.experiments.logging import perform_experiment

data_folder = Path(__file__).parent.parent / 'test_data'
data_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

plot_folder = Path(__file__).parent.parent / 'test_plots'
plot_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

class TestAnalysis(unittest.TestCase):
    def test_plotting_single(self):
        # log data
        env = TwoActionsTwoStates()
        eval_env = TwoActionsTwoStates()
        agent = SAC(env, eval_env=eval_env, lr=0.01, buffer_size=100, batch_size=50, entropy_scale=0.2)

        file_name = 'sac_e'
        if Path.exists(data_folder / f'{file_name}.csv'):
            os.remove(data_folder / f'{file_name}.csv')
        perform_experiment(agent, file_name=file_name, file_path=data_folder, num_steps=10000, track_rate=100,
                           iter_per_track=50)

        # plotting
        stats = analyse_experiments([data_folder / 'sac_e'])
        visualize_statistics({'SAC in E': stats}, plot_folder / 'sac_e')

    def test_plotting_multiple_seed(self):
        # log data
        for seed in range(5):
            env = TwoActionsTwoStates()
            eval_env = TwoActionsTwoStates()
            agent = SAC(env, eval_env=eval_env, seed=seed, lr=0.01, buffer_size=100, batch_size=50, entropy_scale=0.2)

            file_name = f'sac_e_{seed}'
            if Path.exists(data_folder / f'{file_name}.csv'):
                os.remove(data_folder / f'{file_name}.csv')  # remove old file
            perform_experiment(agent, file_name=file_name, file_path=data_folder, num_steps=5000, track_rate=100,
                               iter_per_track=50)

        # plotting
        file_list = [data_folder / f'sac_e_{seed}' for seed in range(5)]
        stats = analyse_experiments(file_list)
        visualize_statistics({'SAC in E': stats}, plot_folder / 'sac_e_multiple_seed')
