import torch


@torch.no_grad()
def moving_average(target_params, current_params, factor):
    """
    Shifts the target_params towards the current_params by the given factor
    according to: target = (1 - factor)*target + factor*current
    """
    for t, c in zip(target_params, current_params):
        t += factor * (c - t)


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
