import torch
import numpy as np


def permutation(set):
    permid=np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i]=set[i][permid]

    return set


def iso_KLD(mu, log_sigma_sq):
    return torch.sum( - 0.5 * (1.0 + log_sigma_sq - mu**2 - torch.exp(log_sigma_sq)), dim=1)


def noniso_logpdf(x, mu_prior, cov_prior):
    # log(p(y)) where p(y) is multivariate gaussian

    deviations = x - mu_prior # 100,3
    cov_inverse = torch.inverse(cov_prior) # 3x3

    return - 0.5 * (float(cov_prior.shape[0]) * np.log(2.*np.pi) +  np.log(np.linalg.det(cov_prior))
                + torch.sum( torch.mm(deviations, cov_inverse) * deviations, dim=1) )


def noniso_KLD(self, mu_prior, cov_prior, mu, log_sigma_sq):
    est_deviation = mu_prior - mu
    cov_inverse = torch.inverse(cov_prior)

    noniso_logpdf_val = torch.sum( torch.mm(est_deviation, cov_inverse) * est_deviation, dim=1) - float(cov_prior.shape[0]) + np.log(np.linalg.det(cov_prior))

    exp_sgm = torch.exp(log_sigma_sq)
    print("exp_sgm ", exp_sgm.shape)

    return 0.5 * ( torch.trace( torch.scan(lambda a, x: torch.mm(cov_inverse, x), tf.matrix_diag(exp_sgm) ) ) + noniso_logpdf_val - torch.sum(log_sigma_sq, dim=1) )


def compute_objYpred_MSE(y_L, y_L_mu):
    return F.mse_loss(y_L, y_L_mu) #default:mean

def cross_entropy(x, y, const = 1e-10):
    return - ( x*torch.log(torch.clamp(y, const, 1.0))+(1.0-x)*torch.log(torch.clamp(1.0-y, const, 1.0)))

def compute_log_lik(x, x_recon):
    return - torch.mean( - torch.sum(cross_entropy(torch.flatten(x), torch.flatten(x_recon))))
