import torch
import numpy as np


def permutation(set):
    permid=np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i]=set[i][permid]
    return set


def iso_KLD(mu, log_sigma_sq):
    return torch.sum( - 0.5 * (1.0 + log_sigma_sq - mu**2 - torch.exp(log_sigma_sq)), dim=1)



def compute_objYpred_MSE(y_L, y_L_mu):
    return F.mse_loss(y_L, y_L_mu) #default:mean


def cross_entropy(x, y, const = 1e-10):
    return - ( x*torch.log(torch.clamp(y, const, 1.0))+(1.0-x)*torch.log(torch.clamp(1.0-y, const, 1.0)))


def compute_log_lik(x, x_recon):
    return - torch.sum(cross_entropy(torch.flatten(x), torch.flatten(x_recon)))
