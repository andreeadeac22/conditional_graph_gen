import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

model = CondJTNNVAE(vocab, args.hidden_size, args.prop_hidden_size, args.latent_size, args.depthT, args.depthG)

model.load_state_dict(torch.load(args.model))
model = model.cuda()

torch.manual_seed(0)

def sampling_unconditional():
    z_tree = torch.randn(1, args.latent_size).cuda()
    z_mol = torch.randn(1, args.latent_size).cuda()

    

for i in xrange(args.nsample):
    print model.sample_prior()
    z_tree = torch.randn(1, self.latent_size).cuda()
    z_mol = torch.randn(1, self.latent_size).cuda()
    return self.decode(z_tree, z_mol, prob_decode)


## unconditional generation
for t in range(10):
    smi = model.sampling_unconditional()
    sample_z=np.random.randn(1, self.dim_z)
    sample_y=np.random.multivariate_normal(self.mu_prior, self.cov_prior, 1)

        sample_smiles=self.beam_search(sample_z, sample_y, k=5)

        return sample_smiles

    print([t, smi, get_property(smi)])

## conditional generation (e.g. MolWt=250)
yid = 0
ytarget = 250.
ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

for t in range(10):
    smi = model.sampling_conditional(yid, ytarget_transform)
    print([t, smi, get_property(smi)])
