import torch


@torch.no_grad()
def moving_average(target_params, current_params, factor):
    for t, c in zip(target_params, current_params):
        t += factor * (c - t)


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
