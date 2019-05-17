import torch
ntrn=300000
ntst=10000

frac_val=0.05 #validation proportion out of the training set
frac=0.5 # labeled out of total (for training/validation)

nL = int(ntrn * frac)
nU = ntrn - nL

print("nL is: ", nL)
print("nU is: ", nU)

nL_trn=int(nL*(1-frac_val))
nL_val=nL-nL_trn

nU_trn=int(nU*(1-frac_val))
nU_val=nU-nU_trn

print("nL_trn: ", nL_trn)
print("nL_val: ", nL_val)
print("nU_trn: ", nU_trn)
print("nU_val: ", nU_val


data_uri='../data/'

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch_mu_prior = torch.tensor([359.2917, 2.9117384, 0.69649607], device = cuda_device)
cov_prior = [[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00], \
    [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
    [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]]

torch_cov_prior = torch.tensor(cov_prior, device = cuda_device)
