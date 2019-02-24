import torch
import numpy as np


def permutation(set):
    permid=np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i]=set[i][permid]

    return set


def iso_KLD(self, mu, log_sigma_sq):
    return torch.sum( - 0.5 * (1.0 + log_sigma_sq - torch.square(mu) - torch.exp(log_sigma_sq) ), dim=1)


def noniso_logpdf(self, x, mu_prior, cov_prior):
    return - 0.5 * (float(cov_prior.shape[0]) * np.log(2.*np.pi) +  np.log(np.linalg.det(cov_prior))
                + torch.sum( torch.mm( torch.matmul( torch.subtract(x, mu_prior), torch.matrix_inverse(cov_prior) ), torch.subtract(x, mu_prior) ), dim=1) )


"""
def noniso_KLD(self, mu, log_sigma_sq):
    return 0.5 * ( torch.trace( torch.scan(lambda a, x: tf.matmul(tf.matrix_inverse(self.tf_cov_prior), x), tf.matrix_diag(tf.exp(log_sigma_sq)) ) )
              + tf.reduce_sum( tf.multiply( tf.matmul( tf.subtract(self.tf_mu_prior, mu), tf.matrix_inverse(self.tf_cov_prior) ), tf.subtract(self.tf_mu_prior, mu) ), 1)
              - float(self.cov_prior.shape[0]) + np.log(np.linalg.det(self.cov_prior)) - tf.reduce_sum(log_sigma_sq, 1) )
"""

def compute_objYpred_MSE(y_L, y_L_mu):
    return F.mse_loss(y_L, y_L_mu) #default:mean

def cross_entropy(self, x, y, const = 1e-10):
    return - ( x*torch.log(torch.clamp(y, const, 1.0))+(1.0-x)*torch.log(torch.clamp(1.0-y, const, 1.0)))

def compute_log_lik(x, x_recon):
    return - torch.mean( - torch.sum(cross_entropy(torch.flatten(x), torch.flatten(x_recon)), 1))
