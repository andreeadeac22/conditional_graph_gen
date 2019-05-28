import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import tensorize


def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi)
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]

    except Exception as e:
        print(e)
        property = 'invalid'
    return property

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    #parser.add_argument('--nsample', type=int, required=True)
    #parser.add_argument('--testX', required=True) # ../data/zinc310k/test
    #parser.add_argument('--testY', required=True) # ../data/zinc310k
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_epoch', type=int, default=0)

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

    parser.add_argument('--infomax_factor_true', type=int, default=1)
    parser.add_argument('--infomax_factor_false', type=int, default=1)
    parser.add_argument('--u_kld_y_factor', type=float, default=1.0)
    parser.add_argument('--ymse_factor', type=int, default=1)


    rgs = parser.parse_args()
    print(args)

    batch_size_L=int(args.batch_size* nL_trn / (nL_trn+nU_trn))
    batch_size_U=int(args.batch_size* nU_trn / (nL_trn+nU_trn))

    print("batch_size_L ", batch_size_L)
    print("batch_size_U ", batch_size_U)

    vocab_file = args.data + "vocab.txt"
    vocab = [x.strip("\r\n ") for x in open(vocab_file)]
    vocab = Vocab(vocab)


    with open(args.data + "train.pickle", "rb") as f:
        data = pickle.load(f)

    trnY_L = list(data['trnY_L'])
    trnY_L = np.stack(trnY_L, axis=0)

    mu_prior= torch.tensor(np.mean(trnY_L,0), dtype = torch.float32, device = cuda_device)
    cov_prior= torch.tensor(np.cov(trnY_L.T), dtype = torch.float32, device = cuda_device)

    print("mu_prior ", mu_prior)
    print("cov_prior ", cov_prior)

    model = CondJTNNVAE(vocab, args, batch_size_L, batch_size_U, mu_prior, cov_prior)
    #model = CondJTNNVAE(vocab, args.hidden_size, args.prop_hidden_size, args.latent_size, args.depthT, args.depthG, args.infomax_true, args.infomax_false, args.u_kld_y, args.ymse_factor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if torch.cuda.is_available():
        model = model.cuda()

    loader = SSMolTreeFolder(args.train, vocab, batch_size_L, batch_size_U, num_workers=4) #make siamese dataloader
    #tst_loader = MolTreeFolder(tstX, vocab, args.batch_size, num_workers=4)


    # property prediction performance
    print('::: property prediction performance')

    prediY = []
    model.eval()
    for batch in loader:
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder, \
            prop_x_batch, prop_x_jtenc_holder, prop_x_mpn_holder, prop_x_jtmpn_holder, props \
                = batch

        y_L_mu, y_L_lsgms = self.predi(*prop_x_jtenc_holder)

        loss, \
        prop_loss, prop_tree_kl, prop_mol_kl, prop_word_acc, prop_topo_acc, prop_assm_acc, prop_log_prior_y, \
        u_loss, tree_kl, mol_kl, u_word_acc, u_topo_acc, u_assm_acc, u_kld_y, \
        objYpred_MSE, d_true_loss, d_fake_loss = model(batch, beta)

    for batch in trn_loader:
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        predi_y_L_mu, predi_y_L_lsgms = model.predi(*x_jtenc_holder)
        predi_y_L_mu = predi_y_L_mu.cpu().detach()
        trnY_hat_val = scaler_Y.inverse_transform(predi_y_L_mu.numpy())
        print("trnY_hat_val ", trnY_hat_val)
        print("prediY ", predi_y_L_mu)
        print("trnY_L ", trnY_L)

        dim_y = 3
        with open(args.save_dir + "trn_prop_pred.txt", "w") as e:
            for j in range(dim_y):
                print([j, mean_absolute_error(trnY_L[:,j], predi_y_L_mu[:,j])], file=e)

    for batch in tst_loader:
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        predi_y_L_mu, predi_y_L_lsgms = model.predi(*x_jtenc_holder)
        predi_y_L_mu = predi_y_L_mu.cpu().detach()
        tstY_hat_val = scaler_Y.inverse_transform(predi_y_L_mu.numpy())
        print("tstY_hat_val ", tstY_hat_val)
        print("prediY ", predi_y_L_mu)
        print("tstY ", tstY)

        dim_y = 3
        with open(args.save_dir + "tst_prop_pred.txt", "w") as e:
            for j in range(dim_y):
                print([j, mean_absolute_error(tstY[:,j], predi_y_L_mu[:,j])], file=e)
