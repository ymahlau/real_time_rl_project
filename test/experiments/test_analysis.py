import unittest
from pathlib import Path

from src.experiments.analysis import analyse_experiments, visualize_statistics

data_folder = Path(__file__).parent.parent / 'test_data'
data_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

plot_folder = Path(__file__).parent.parent / 'test_plots'
plot_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

class TestAnalysis(unittest.TestCase):
    def test_plotting_single(self):
        stats = analyse_experiments([data_folder / 'sac_e'])
        visualize_statistics({'SAC in E': stats}, plot_folder / 'sac_e')

    def test_plotting_multiple_seed(self):
        file_list = [data_folder / f'sac_e_{seed}' for seed in range(5)]
        stats = analyse_experiments(file_list)
        visualize_statistics({'SAC in E': stats}, plot_folder / 'sac_e_multiple_seed')
