import torch

##### conditional constants
num_prop = 3
cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch_mu_prior = torch.tensor([359.2917, 2.9117384, 0.69649607], device = cuda_device)
torch_cov_prior = torch.tensor([[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00],\
    [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
    [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]], device=cuda_device)

print(torch_cov_prior)
