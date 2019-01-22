import torch
import torch.nn.functional as F

def _permutation(self, set):

    permid=np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i]=set[i][permid]

    return set


def compute_objYpred_MSE(y_L, y_L_mu):
    return F.mse_loss(y_L, y_L_mu) #default:mean

def cross_entropy(self, x, y, const = 1e-10):
    return - ( x*torch.log(torch.clamp(y, const, 1.0))+(1.0-x)*torch.log(torch.clamp(1.0-y, const, 1.0)))

def compute_obj_val(x, x_recon):
    return - torch.mean( - torch.sum(cross_entropy(torch.flatten(x), torch.flatten(x_recon)), 1))
