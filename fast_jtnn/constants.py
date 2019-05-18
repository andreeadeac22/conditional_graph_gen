import torch

##### conditional constants
num_prop = 3
cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print("nU_val: ", nU_val)


data_uri= '../data/'
