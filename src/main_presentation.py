from pathlib import Path

from src.experiments.analysis import analyse_experiments, visualize_statistics

img_folder = Path(__file__).parent.parent / 'plots'
img_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

data_folder = Path(__file__).parent.parent / 'experiment_data'


def vanilla_plots(img_counter: int):
    img_counter += 1
    sac_names = [f'SAC-S{seed}-T200' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T200-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T200-rtmdp' for seed in range(5)]

    sac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_names])
    sac_rtmdp_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_rtmdp_names])
    rtac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in rtac_names])
    visualize_statistics({'SAC in E': sac_stats_acrobot, 'RTAC': rtac_stats_acrobot,
                          'SAC in RTMDP': sac_rtmdp_stats_acrobot},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_acrobot',
                         x_lim=(10000, 100000), smoothing_factor=2, show=False)

    img_counter += 1
    sac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_names])
    sac_rtmdp_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_rtmdp_names])
    rtac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in rtac_names])
    visualize_statistics({'SAC in E': sac_stats_cartpole, 'RTAC': rtac_stats_cartpole,
                          'SAC in RTMDP': sac_rtmdp_stats_cartpole},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_cartpole',
                         x_lim=(10000, 100000), smoothing_factor=10, show=False)

    img_counter += 1
    sac_names = [f'SAC-S{seed}-T600' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T600-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T600-rtmdp' for seed in range(5)]
    sac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_names])
    sac_rtmdp_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_rtmdp_names])
    rtac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in rtac_names])
    visualize_statistics({'SAC in E': sac_stats_lunar, 'RTAC': rtac_stats_lunar,
                          'SAC in RTMDP': sac_rtmdp_stats_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_lunar', y_lim=(-200, 200),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
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
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_lunar, 'SAC RTMDP + Target': sac_target_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_target_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_lunar, 'RTAC + Target': rtac_target_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_target_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_cart, 'SAC RTMDP + Target': sac_target_cart},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_target_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_cart, 'RTAC + Target': rtac_target_cart},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_target_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)

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
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_lunar, 'SAC RTMDP + Double Value': sac_double_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_double_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_lunar, 'RTAC + Double Value': rtac_double_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_double_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_cart, 'SAC RTMDP + Double Value': sac_double_cart},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_double_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_cart, 'RTAC + Double Value': rtac_double_cart},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_double_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)

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
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_lunar, 'SAC RTMDP + Norm': sac_norm_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_norm_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_lunar, 'RTAC + Norm': rtac_norm_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'SAC RTMDP Vanilla': sac_vanilla_cart, 'SAC RTMDP + Norm': sac_norm_cart},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_norm_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_cart, 'RTAC + Norm': rtac_norm_cart},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_norm_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)

    # shared data
    rtac_shared_lunar = analyse_experiments([data_folder / 'LunarLander' / f'RTAC-S{s}-T600-shared-rtmdp'
                                             for s in range(5)])
    rtac_shared_cart = analyse_experiments([data_folder / 'CartPole' / f'RTAC-S{s}-T200-shared-rtmdp'
                                            for s in range(5)])

    # shared plots
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_lunar, 'RTAC + Shared Params': rtac_shared_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_shared_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla_cart, 'RTAC + Shared Params': rtac_shared_cart},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_shared_cart', y_lim=(-200, 600),
                         x_lim=(10000, 100000), smoothing_factor=5, show=False)

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
    visualize_statistics({'SAC Vanilla': sac_vanilla, 'SAC + All': sac_all},
                         save_dest=img_folder / f'{img_counter:02d}_sac_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'SAC Vanilla RTMDP': sac_rtmdp_vanilla, 'SAC RTMDP + All': sac_rtmdp_all},
                         save_dest=img_folder / f'{img_counter:02d}_sac_rtmdp_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)
    img_counter += 1
    visualize_statistics({'RTAC Vanilla': rtac_vanilla, 'RTAC + All': rtac_all},
                         save_dest=img_folder / f'{img_counter:02d}_rtac_all_lunar', y_lim=(-200, 250),
                         x_lim=(10000, 300000), smoothing_factor=10, show=False)

    return img_counter




def main_presentation():
    # 1 plots of vanilla sac, rtac, sac in rtmdp
    img_counter = 0
    # img_counter = vanilla_plots(img_counter)

    # 2 Extensions Single
    img_counter = 3
    # img_counter = extensions_single(img_counter)

    # 3 Combinations of Extensions
    img_counter = 17
    img_counter = extensions_combinations(img_counter)


if __name__ == '__main__':
    main_presentation()
