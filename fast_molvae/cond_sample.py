import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

from fast_jtnn import *

def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi)
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]

    except:
        property = 'invalid'

    return property

if __name__ == "__main__":
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

    if torch.cuda.is_available():
        model = model.cuda()

    f = open(path + "test.pickle", "rb")
    data = pickle.load(f)
    tstX_L = list(data['tstX_L'])
    tstY_L = list(data['tstY_L'])
    tstX_U = list(data['tstX_U'])
    # model testing
    print('::: model testing')

    # property prediction performance
    print('::: property prediction performance')
    tstX =  torch.tensor(tstX, dtype=torch.float32, device=device)
    predi_y_L_mu, predi_y_L_lsgms = model.predi(tstX)
    tstY_hat = scaler_Y.inverse_transform(predi_y_L_mu.cpu().detach().numpy())
    with open("prop_pred.txt", "w") as e:
        for j in range(dim_y):
            print([j, mean_absolute_error(tstY_L[:,j], tstY_hat[:,j])], file=e)

    ## unconditional generation
    print('::: unconditional generation')
    with open("uncond_gen.txt", "w") as f:
        for t in range(10):
            smi = model.sampling_unconditional()
            print([t, smi, get_property(smi)], file=f)

    ## conditional generation (e.g. MolWt=250)
    print('::: conditional generation (e.g. MolWt=250)')
    yid = 0
    ytarget = 250.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    with open("cond_gen_molwt250.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)
