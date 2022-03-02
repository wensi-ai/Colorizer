import torch

def mse_loss(input, target=0):
    return torch.mean((input - target) ** 2)


def l1_loss(input, target=0):
    return torch.mean(torch.abs(input - target))


def weighted_mse_loss(input, target, weights):
    out = (input - target) ** 2
    out = out * weights.expand_as(out)
    return out.mean()


def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    return out.mean()