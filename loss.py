import torch


def kl_divergence(mu: torch.tensor, log_var: torch.tensor):
    # output = -1/2 (log_var + 1 -torch.add(torch.pow(mu, 2), torch.exp(log_var)))
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    return torch.sum(KLD_element).mul_(-0.5)
