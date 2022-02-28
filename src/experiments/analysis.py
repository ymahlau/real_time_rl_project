from pathlib import Path
from typing import List, Any, Union, Dict, Optional, Tuple
import numpy as np
from scipy.stats import bootstrap
from matplotlib import pyplot as plt


def smooth(array: np.ndarray, smoothing_factor: int) -> np.ndarray:
    """
    Smooths a 1-D input array by taking the average of the smoothing_factor neighboring points for every entry in the
    array.
    """
    n = len(array)
    indices = np.arange(0, n)
    lower_idx = np.maximum(indices - int(smoothing_factor / 2), 0)
    upper_idx = np.minimum(indices + int(smoothing_factor / 2), n-1)

    x = [
        np.mean(array[lower_idx[i]:upper_idx[i]]).item()
        for i
        in indices
    ]
    result = np.asarray(x)
    return result

def visualize_statistics(
        statistics: Dict[str, np.ndarray],
        save_dest: Optional[Union[str, Path]] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        smoothing_factor: int = 1,
):

    """
    Plots given analysed experiment data (statistics) and saves the plotted data.
    x-axis represents the training steps and y-axis represents the mean and confidence interval
    of average return for current training step.

    statistics: Experiment statistics which are computed in analyse_experiment.
    save_dest: Path to where the plot is to be saved.
    x_lim: The limit of x-axis in the plot.
    y_lim: The limit of y-axis in the plot.
    smoothing_factor: How many points to average to single new point.
    """

    #Setup plot
    plt.clf()
    plt.figure(figsize=(8, 8), dpi=200)
    plt.xlabel("Steps")
    plt.ylabel("Average return")

    #Fill plot with the given statistics
    for name, stats in statistics.items():
        x = stats[:, 0]
        y = smooth(stats[:, 1], smoothing_factor=smoothing_factor)
        y_lower = smooth(stats[:, 2], smoothing_factor=smoothing_factor)
        y_upper = smooth(stats[:, 3], smoothing_factor=smoothing_factor)

        plt.plot(x, y, label=name)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    #Show and save plot
    plt.legend()
    if save_dest is not None:
        plt.savefig(f"{save_dest}.png")
    plt.show()


def analyse_experiments(data_paths: List[Union[str, Path]]) -> np.ndarray:
    """"
    Computes the means and confidence intervals of the data points for a given experiment.
    data_paths: Paths to the files containing the experiment data. The data paths should refer
    to csv files created by logging.perform_experiment.

    returns: A list containing tuples of current step, mean at that step, upper bound, lower bound (of confidence
    interval of average return at that step) of experiment data
    """
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

        #Check data consistency
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

def total_regret(stats: np.ndarray, max_return: float) -> Tuple[float, float, float]:
    """
    Computes the total regret of a given statistics array.
    stats: the stats array should be computed using the analyse_experiments function.
    max_return: the maximum possible return of a single episode

    returns:
    (mean total regret,
    lower confidence bound of total regret,
    upper confidence bound of total regret)
    """
    y = stats[:, 1]
    y_lower = stats[:, 2]
    y_upper = stats[:, 3]

    avg_regret = np.sum(np.clip(max_return - y, 0, None)).item()
    lower_regret = np.sum(np.clip(max_return - y_upper, 0, None)).item()
    upper_regret = np.sum(np.clip(max_return - y_lower, 0, None)).item()

    return avg_regret, lower_regret, upper_regret





"""
Usage Example:

stats = analyse_experiments(["experiment_data/cartpole_test_data/cartpole{0}".format(i) for i in range(9)])
stats_with_target = analyse_experiments(["experiment_data/cartpole_test_data/cartpole_with_target{0}".format(i) for i in range(9)])
visualize_statistics([stats,stats_with_target], "experiment_data/cartpole_test_data/bild")
"""
