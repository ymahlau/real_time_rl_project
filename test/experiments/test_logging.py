import os
import unittest
from pathlib import Path

from src.agents.sac import SAC
from src.envs.probe_envs import TwoActionsTwoStates
from src.experiments.logging import perform_experiment

data_folder = Path(__file__).parent.parent / 'test_data'
data_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

plot_folder = Path(__file__).parent.parent / 'test_plots'
plot_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

class TestLogging(unittest.TestCase):
    def test_logging_sac_e_single(self):
        env = TwoActionsTwoStates()
        eval_env = TwoActionsTwoStates()
        agent = SAC(env, eval_env=eval_env, lr=0.01, buffer_size=100, batch_size=50, entropy_scale=0.2)

        file_name = 'sac_e'
        os.remove(data_folder / f'{file_name}.csv')
        perform_experiment(agent, file_name=file_name, file_path=data_folder, num_steps=10000, track_rate=100,
                           iter_per_track=50)

    def test_logging_multiple_seed(self):
        for seed in range(5):
            env = TwoActionsTwoStates()
            eval_env = TwoActionsTwoStates()
            agent = SAC(env, eval_env=eval_env, seed=seed, lr=0.01, buffer_size=100, batch_size=50, entropy_scale=0.2)

            file_name = f'sac_e_{seed}'
            os.remove(data_folder / f'{file_name}.csv')  # remove old file
            perform_experiment(agent, file_name=file_name, file_path=data_folder, num_steps=5000, track_rate=100,
                               iter_per_track=50)





