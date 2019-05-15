import torch
import torch.nn as nn
import pickle
import numpy

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fast_molvae_constants import *
from preprocess import tensorize


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
    #parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--testX', required=True) # ../data/zinc310k/test
    parser.add_argument('--testY', required=True) # ../data/zinc310k
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--prop_hidden_size', type=int, default=3) #cond
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.001)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=40000)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=40000)
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=5000)

    parser.add_argument('--infomax_true', type=int, default=1)
    parser.add_argument('--infomax_false', type=int, default=1)
    parser.add_argument('--u_kld_y', type=int, default=1)
    parser.add_argument('--ymse_factor', type=int, default=1)

    args = parser.parse_args()

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    tstY = torch.tensor(pickle.load(open(args.testY + "processed-testY.pkl", "rb")))

    """
    f = open(args.testX + "processed-test.pkl", "rb")
    data = pickle.load(f)
    tstX = list(data['tstX'])
    tstY = list(data['tstY'])

    with open(args.testX + "processed-testX.pkl", "wb") as g:
        pickle.dump(tstX, g)

    with open(args.testY + "processed-testY.pkl", "wb") as h:
        pickle.dump(tstY, h)
    print("tstX ", tstX[0])
    """

    """
    tstX = [tensorize(smile) for smile in tstX]
    testdict = {}
    testdict['tstX'] = tstX
    testdict['tstY'] = tstY
    with open(args.test + "processed-test.pkl", "wb") as g:
        pickle.dump(testdict, g)
    """

    model = CondJTNNVAE(vocab, args.hidden_size, args.prop_hidden_size, args.latent_size, args.depthT, args.depthG, args.infomax_true, args.infomax_false, args.u_kld_y, args.ymse_factor)

    state = torch.load(args.model) # need to change
    model.load_state_dict(state['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    loader = MolTreeFolder(args.testX, vocab, args.batch_size, num_workers=4)

    # model testing
    print('::: model testing')

    # property prediction performance
    print('::: property prediction performance')
    #tstX =  torch.tensor(tstX, dtype=torch.float32, device=cuda_device)
    #tstY_hat = []
    prediY = []
    #it = iter(loader)
    nb = 0
    for batch in loader:
        #if nb > 3:
        #    break

        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        predi_y_L_mu, predi_y_L_lsgms = model.predi(*x_jtenc_holder)
        #tstY_hat_val = scaler_Y.inverse_transform(predi_y_L_mu.cpu().detach().numpy())
        #print("tstY_hat_val ", tstY_hat_val)
        #print("prediY", predi_y_L_mu)
        #tstY_hat += [tstY_hat_val]
        prediY += [predi_y_L_mu]

        nb +=1

    prediY = torch.cat(prediY)
    print(prediY.shape)
    prediY = prediY.detach().numpy()

    dim_y = 3
    with open("prop_pred.txt", "w") as e:
        for j in range(dim_y):
            print([j, mean_absolute_error(tstY[:64,j], prediY[:,j])], file=e)

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
    #ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    with open("cond_gen_molwt250.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget)
            print([t, smi, get_property(smi)], file=g)
