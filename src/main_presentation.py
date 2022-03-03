from pathlib import Path

import numpy as np

from src.experiments.analysis import analyse_experiments, visualize_statistics, total_regret

img_folder = Path(__file__).parent.parent / 'plots'
img_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

data_folder = Path(__file__).parent.parent / 'experiment_data'

rtac_color = 'orange'
sac_color = 'blue'
sac_rtmdp_color = 'green'
ext_color = 'black'

def vanilla_plots(img_counter: int):
    img_counter += 1
    sac_names = [f'SAC-S{seed}-T200' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T200-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T200-rtmdp' for seed in range(5)]

    sac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_names])
    sac_rtmdp_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_rtmdp_names])
    rtac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in rtac_names])
    visualize_statistics({'SAC in E': (sac_stats_acrobot, sac_color), 'RTAC': (rtac_stats_acrobot, rtac_color),
                          'SAC in RTMDP': (sac_rtmdp_stats_acrobot, sac_rtmdp_color)},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_acrobot',
                         x_lim=(10000, 30000), smoothing_factor=2, show=False)

    img_counter += 1
    sac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_names])
    sac_rtmdp_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_rtmdp_names])
    rtac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in rtac_names])
    visualize_statistics({'SAC in E': (sac_stats_cartpole, sac_color), 'RTAC': (rtac_stats_cartpole, rtac_color),
                          'SAC in RTMDP': (sac_rtmdp_stats_cartpole, sac_rtmdp_color)},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_cartpole',
                         x_lim=(10000, 100000), smoothing_factor=50, show=False)

    img_counter += 1
    sac_names = [f'SAC-S{seed}-T600' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T600-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T600-rtmdp' for seed in range(5)]
    sac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_names])
    sac_rtmdp_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_rtmdp_names])
    rtac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in rtac_names])
    visualize_statistics({'SAC in E': (sac_stats_lunar, sac_color), 'RTAC': (rtac_stats_lunar, rtac_color),
                          'SAC in RTMDP': (sac_rtmdp_stats_lunar, sac_rtmdp_color)},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_lunar', y_lim=(-200, 200),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    return img_counter


def extensions_single(img_counter):
    # vanilla stats
    sac_vanilla_lunar = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-rtmdp' for s in range(5)])
    sac_vanilla_cart = analyse_experiments([data_folder / 'CartPole' / f'SAC-S{s}-T200-rtmdp' for s in range(5)])
    rtac_vanilla_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-rtmdp' for s in range(5)])
    rtac_vanilla_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-rtmdp' for s in range(5)])

    # target data
    sac_target_lunar = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-target-rtmdp'
                                            for s in range(5)])
    sac_target_cart = analyse_experiments([data_folder / 'CartPole' / f'SAC-S{s}-T200-target-rtmdp'
                                           for s in range(5)])
    rtac_target_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-target-rtmdp'
                                             for s in range(5)])
    rtac_target_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-target-rtmdp'
                                            for s in range(5)])

    # target plots
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_lunar, sac_rtmdp_color),
                          'SAC RTMDP + Target': (sac_target_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_target_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_lunar, rtac_color), 'RTAC + Target': (rtac_target_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_target_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_cart, sac_rtmdp_color), 'SAC RTMDP + Target': (sac_target_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_target_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_cart, rtac_color), 'RTAC + Target': (rtac_target_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_target_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)

    # double value data
    sac_double_lunar = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-double-rtmdp'
                                            for s in range(5)])
    sac_double_cart = analyse_experiments([data_folder / 'CartPole' / f'SAC-S{s}-T200-double-rtmdp'
                                           for s in range(5)])
    rtac_double_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-double-rtmdp'
                                             for s in range(5)])
    rtac_double_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-double-rtmdp'
                                            for s in range(5)])

    # double value plots
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_lunar, sac_rtmdp_color), 'SAC RTMDP + Double Value': (sac_double_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_double_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_lunar, rtac_color), 'RTAC + Double Value': (rtac_double_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_double_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_cart, sac_rtmdp_color), 'SAC RTMDP + Double Value': (sac_double_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_double_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_cart, rtac_color), 'RTAC + Double Value': (rtac_double_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_double_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)

    # normalization data
    sac_norm_lunar = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-norm-rtmdp'
                                          for s in range(5)])
    sac_norm_cart = analyse_experiments([data_folder / 'CartPole' / f'SAC-S{s}-T200-norm-rtmdp'
                                         for s in range(5)])
    rtac_norm_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-norm-rtmdp'
                                           for s in range(5)])
    rtac_norm_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-norm-rtmdp'
                                          for s in range(5)])

    # normalization plots
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_lunar, sac_rtmdp_color), 'SAC RTMDP + Norm': (sac_norm_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_norm_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_lunar, rtac_color), 'RTAC + Norm': (rtac_norm_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': (sac_vanilla_cart, sac_rtmdp_color), 'SAC RTMDP + Norm': (sac_norm_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_norm_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_cart, rtac_color), 'RTAC + Norm': (rtac_norm_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)

    # shared data
    rtac_shared_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-shared-rtmdp'
                                             for s in range(5)])
    rtac_shared_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-shared-rtmdp'
                                            for s in range(5)])

    # shared plots
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_lunar, rtac_color), 'RTAC + Shared Params': (rtac_shared_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_shared_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_cart, rtac_color), 'RTAC + Shared Params': (rtac_shared_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_shared_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=20, show=False)

    return img_counter


def extensions_combinations(img_counter):
    # vanilla stats
    sac_rtmdp_vanilla = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-rtmdp' for s in range(5)])
    sac_vanilla = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600' for s in range(5)])
    rtac_vanilla = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-rtmdp' for s in range(5)])

    # combination stats
    sac_rtmdp_all = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-target-double-norm-rtmdp'
                                         for s in range(5)])
    sac_all = analyse_experiments([data_folder / 'LunarLander' / f'SAC-S{s}-T600-target-double-norm' for s in range(5)])
    rtac_all = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-target-double-norm-shared-rtmdp'
                                    for s in range(5)])

    # plots
    img_counter += 1
    visualize_statistics({'SAC Vanilla': (sac_vanilla, sac_color), 'SAC + All': (sac_all, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'SAC Vanilla RTMDP': (sac_rtmdp_vanilla, sac_rtmdp_color), 'SAC RTMDP + All': (sac_rtmdp_all, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla, rtac_color), 'RTAC + All': (rtac_all, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=50, show=False)

    return img_counter


def shared_norm(img_counter: int):
    # vanilla
    rtac_vanilla_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-rtmdp' for s in range(5)])
    rtac_vanilla_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-rtmdp' for s in range(5)])

    rtac_comb_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-norm-shared-rtmdp'
                                           for s in range(5)])
    rtac_comb_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-norm-shared-rtmdp'
                                          for s in range(5)])

    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_lunar, rtac_color), 'RTAC Norm + Shared': (rtac_comb_lunar, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_shared_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=1, show=False)

    img_counter += 1
    visualize_statistics({'RTAC Vanilla': (rtac_vanilla_cart, rtac_color), 'RTAC Norm + Shared': (rtac_comb_cart, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_shared_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=1, show=False)

    return img_counter


def experiment_step_size(img_counter: int):
    step_sizes = [0.01, 0.03, 0.1, 0.3, 1]
    max_return = 200
    result_rtac = np.zeros(shape=(4, 5))
    result_sac = np.zeros(shape=(4, 5))
    result_sac_rtmdp = np.zeros(shape=(4, 5))
    result_sac_prev = np.zeros(shape=(4, 5))

    for i, step_size in enumerate(step_sizes):
        stats_rtac = analyse_experiments([data_folder / 'CustomLunarLander' / f'RTAC-S{s}-T100-rtmdp-{step_size}'
                                          for s in range(3)])
        stats_sac = analyse_experiments([data_folder / 'CustomLunarLander' / f'SAC-S{s}-T100-{step_size}'
                                         for s in range(3)])
        stats_sac_rtmdp = analyse_experiments([data_folder / 'CustomLunarLander' / f'SAC-S{s}-T100-rtmdp-{step_size}'
                                               for s in range(3)])
        stats_sac_prev = analyse_experiments([data_folder / 'CustomLunarLander' / f'SAC-S{s}-T100-{step_size}-prev'
                                               for s in range(3)])

        regret_rtac = total_regret(stats_rtac, max_return=max_return)
        regret_sac = total_regret(stats_sac, max_return=max_return)
        regret_sac_rtmdp = total_regret(stats_sac_rtmdp, max_return=max_return)
        stats_sac_prev = total_regret(stats_sac_prev, max_return=max_return)

        result_rtac[0, i] = step_size
        result_rtac[1, i] = regret_rtac[0]
        result_rtac[2, i] = regret_rtac[1]
        result_rtac[3, i] = regret_rtac[2]

        result_sac[0, i] = step_size
        result_sac[1, i] = regret_sac[0]
        result_sac[2, i] = regret_sac[1]
        result_sac[3, i] = regret_sac[2]

        result_sac_rtmdp[0, i] = step_size
        result_sac_rtmdp[1, i] = regret_sac_rtmdp[0]
        result_sac_rtmdp[2, i] = regret_sac_rtmdp[1]
        result_sac_rtmdp[3, i] = regret_sac_rtmdp[2]

        result_sac_prev[0, i] = step_size
        result_sac_prev[1, i] = stats_sac_prev[0]
        result_sac_prev[2, i] = stats_sac_prev[1]
        result_sac_prev[3, i] = stats_sac_prev[2]

    img_counter += 1
    visualize_statistics({'RTAC': (result_rtac.T, rtac_color), 'SAC': (result_sac.T, sac_color),
                          'SAC in RTMDP': (result_sac_rtmdp.T, sac_rtmdp_color),
                          'SAC with prev': (result_sac_prev.T, ext_color)},
                         save_dest=img_folder / f'{img_counter:02d}_step_sizes', log=True, y_name='Total Regret',
                         x_name='Step Size')

    return img_counter






def main_presentation():
    # 1 plots of vanilla sac, rtac, sac in rtmdp
    img_counter = 0
    img_counter = vanilla_plots(img_counter)

    # 2 Extensions Single
    img_counter = 3
    img_counter = extensions_single(img_counter)

    # 3 Combinations of Extensions
    img_counter = 17
    img_counter = extensions_combinations(img_counter)

    # 3 Norm + Shared
    img_counter = 20
    img_counter = shared_norm(img_counter)

    img_counter = 22
    img_counter = experiment_step_size(img_counter)


if __name__ == '__main__':
    main_presentation()
