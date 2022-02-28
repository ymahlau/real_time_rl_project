from pathlib import Path

from src.experiments.analysis import analyse_experiments, visualize_statistics

img_folder = Path(__file__).parent.parent / 'plots'
img_folder.mkdir(parents=True, exist_ok=True)  # create folder if not existing

data_folder = Path(__file__).parent.parent / 'experiment_data'

def main_presentation():
    img_counter = 0

    # 1 plots of vanilla sac, rtac, sac in rtmdp
    img_counter += 1
    sac_names = [f'SAC-S{seed}-T200' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T200-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T200-rtmdp' for seed in range(5)]

    sac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_names])
    sac_rtmdp_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in sac_rtmdp_names])
    rtac_stats_acrobot = analyse_experiments([data_folder / 'Acrobot' / name for name in rtac_names])
    visualize_statistics({'SAC': sac_stats_acrobot, 'RTAC': rtac_stats_acrobot, 'SAC in E': sac_rtmdp_stats_acrobot},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_acrobot')

    img_counter += 1
    sac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_names])
    sac_rtmdp_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in sac_rtmdp_names])
    rtac_stats_cartpole = analyse_experiments([data_folder / 'CartPole' / name for name in rtac_names])
    visualize_statistics({'SAC': sac_stats_cartpole, 'RTAC': rtac_stats_cartpole, 'SAC in E': sac_rtmdp_stats_cartpole},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_cartpole')

    img_counter += 1
    sac_names = [f'SAC-S{seed}-T600' for seed in range(5)]
    sac_rtmdp_names = [f'SAC-S{seed}-T600-rtmdp' for seed in range(5)]
    rtac_names = [f'RTAC-S{seed}-T600-rtmdp' for seed in range(5)]
    sac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_names])
    sac_rtmdp_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in sac_rtmdp_names])
    rtac_stats_lunar = analyse_experiments([data_folder / 'LunarLander' / name for name in rtac_names])
    visualize_statistics({'SAC': sac_stats_lunar, 'RTAC': rtac_stats_lunar, 'SAC in E': sac_rtmdp_stats_lunar},
                         save_dest=img_folder / f'{img_counter:02d}_vanilla_cartpole', y_lim=(-1000, 200))

    # 2 Extensions




if __name__ == '__main__':
    main_presentation()
