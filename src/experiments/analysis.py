from pathlib import Path
from typing import List, Any, Union, Dict
import numpy as np
from scipy.stats import bootstrap
from matplotlib import pyplot as plt


def visualize_statistics(statistics: Dict[str, np.ndarray], save_dest: Union[str, Path]):
    plt.clf()
    plt.figure(figsize=(8, 8), dpi=200)
    plt.xlabel("Steps")
    plt.ylabel("Average return")

    for name, stats in statistics.items():
        x = stats[:, 0]
        y = stats[:, 1]
        y_lower = stats[:, 2]
        y_upper = stats[:, 3]

        plt.plot(x, y, label=name)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1)

    plt.legend()
    plt.savefig(f"{save_dest}.png")
    plt.show()


def analyse_experiments(data_paths: List[Union[str, Path]]) -> np.ndarray:
    # Read experiment data from files
    data = []
    num_data_points = None
    for data_path in data_paths:
        data_single_exp = []
        f = open(f"{data_path}.csv", "r")
        f.readline()  # skip header
        for line in f:
            split = line.strip().split(",")
            data_single_exp.append([int(split[0]), float(split[1])])
        f.close()
        if num_data_points is None:
            num_data_points = len(data_single_exp)
        if num_data_points != len(data_single_exp):
            raise ValueError(
                f"Data format is invalid: The amount of data points in {data_path}.csv "
                f"differs from {data_paths[0]}.csv")

        data.append(np.asarray(data_single_exp))
    data = np.asarray(data)

    # calculate mean and confidence intervals
    statistics = []
    for i in range(len(data[0])):
        avg_rewards = [data[:, i, 1]]
        if [data[:, i, 0][0]] * (len(data[:, i, 0])) != list(data[:, i, 0]):
            raise ValueError(f"Data format is invalid: The row {i} contains varying step counters.")
        num_steps = data[:, i, 0][0]
        mean = np.mean(avg_rewards)
        if np.std(avg_rewards) == 0:
            statistics.append([num_steps, mean, mean, mean])
        else:
            conf_interval = bootstrap(avg_rewards, np.mean, confidence_level=0.95).confidence_interval
            statistics.append([num_steps, mean, conf_interval.low, conf_interval.high])
    statistics = np.asarray(statistics)

    return statistics


"""
Usage Example:

stats = analyse_experiments(["experiment_data/cartpole_test_data/cartpole{0}".format(i) for i in range(9)])
stats_with_target = analyse_experiments(["experiment_data/cartpole_test_data/cartpole_with_target{0}".format(i) for i in range(9)])
visualize_statistics([stats,stats_with_target], "experiment_data/cartpole_test_data/bild")
"""
