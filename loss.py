import torch


def kl_divergence(mu: torch.tensor, log_var: torch.tensor):
    kl_div_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    return torch.mean(kl_div_element).mul_(-0.5)
