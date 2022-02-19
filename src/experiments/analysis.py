from typing import List, Any
import numpy as np
from scipy.stats import bootstrap
from matplotlib import pyplot as plt


def visualize_statistics(statistics: List, save_dest: str):
    plt.clf()
    plt.xlabel("Steps")
    plt.ylabel("Average return")

    for i in range(len(statistics)):
        x = statistics[i][:, 0]
        y = statistics[i][:, 1]
        y_lower = statistics[i][:, 2]
        y_upper = statistics[i][:, 3]

        plt.plot(x, y)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1)

    plt.savefig("{0}.png".format(save_dest))


def analyse_experiments(data_paths: List[str]) -> np.ndarray:
    # Read experiment data from files
    data = []
    num_data_points = None
    for data_path in data_paths:
        data_single_exp = []
        f = open("{0}.csv".format(data_path), "r")
        f.readline()  # skip header
        for line in f:
            split = line.strip().split(",")
            data_single_exp.append([int(split[0]), float(split[1])])
        f.close()
        if num_data_points is None:
            num_data_points = len(data_single_exp)
        if num_data_points != len(data_single_exp):
            raise ValueError(
                "Data format is invalid: The amount of data points in {0}.csv differs from {1}.csv".format(data_path,
                                                                                                           data_paths[
                                                                                                               0]))
        data.append(np.asarray(data_single_exp))
    data = np.asarray(data)

    # calculate mean and confidence intervals
    statistics = []
    for i in range(len(data[0])):
        avg_rewards = [data[:, i, 1]]
        if [data[:, i, 0][0]] * (len(data[:, i, 0])) != list(data[:, i, 0]):
            raise ValueError("Data format is invalid: The row {0} contains varying step counters.".format(i))
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
