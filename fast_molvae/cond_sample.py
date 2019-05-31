import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

#from fast_molvae_constants import *
from preprocess import tensorize


def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi)
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]

    except Exception as e:
        print(e)
        property = 'invalid'
    return property


def create_test_files(args):
    tst = pickle.load(open(args.data + "test.pickle", "rb"))

    tstX = [tensorize(smile) for smile in tst['tstX']]
    tstY = list(tst['tstY'])

    with open(args.testX + "processed-testX.pkl", "wb") as g:
        pickle.dump(tstX, g)

    with open(args.testY + "processed-testY.pkl", "wb") as h:
        pickle.dump(tstY, h)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    #parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--testX', required=True) # ../data/zinc310k/test
    parser.add_argument('--testY', required=True) # ../data/zinc310k
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--data', required=True)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--prop_hidden_size', type=int, default=3) #cond
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--u_kld_y_factor', type=int, default=1)
    parser.add_argument('--ymse_factor', type=int, default=1)

    args = parser.parse_args()

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    #create_test_files(args)
    #exit(0)

    tstY = np.asarray(pickle.load(open(args.testY + "processed-testY.pkl", "rb")))

    batch_size_L= 16
    batch_size_U= 16

    with open(args.data + "train.pickle", "rb") as f:
        data = pickle.load(f)

    trnY_L = list(data['trnY_L'])
    trnY_L = np.stack(trnY_L, axis=0)

    mu_prior= torch.tensor(np.mean(trnY_L,0), dtype = torch.float32, device = cuda_device)
    cov_prior= torch.tensor(np.cov(trnY_L.T), dtype = torch.float32, device = cuda_device)


    model = CondJTNNVAE(vocab, args, batch_size_L, batch_size_U, mu_prior, cov_prior)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    if torch.cuda.is_available():
        model = model.cuda()

    loader = MolTreeFolder(args.testX, vocab, args.batch_size, num_workers=4)

    # model testing
    print('::: model testing')

    # property prediction performance
    print('::: property prediction performance')
    #tstX =  torch.tensor(tstX, dtype=torch.float32, device=cuda_device)


    file_name = data_uri + "zinc310k/ZINC_310k.csv"
    X = pd.read_csv(file_name).values[:ntrn+ntst,0] #0: SMILES
    Y = np.asarray(pd.read_csv(file_name).values[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED
    Y=Y[:ntrn]
    scaler_Y = StandardScaler()
    scaler_Y.fit(Y)

    scaled_tstY=scaler_Y.transform(tstY)

    prediY = []
    tstY_hat = []
    model.eval()
    #nb=0

    for batch in loader:
        #if nb>3:
        #    break
        #nb+=1
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        predi_y_L_mu, predi_y_L_lsgms = model.predi(*x_jtenc_holder)
        tstY_hat_val = scaler_Y.inverse_transform(predi_y_L_mu.cpu().detach().numpy())
        #print("tstY_hat_val ", tstY_hat_val)
        #print("prediY", predi_y_L_mu)
        tstY_hat += [tstY_hat_val]
        prediY += [predi_y_L_mu.detach()]


    prediY = torch.cat(prediY)
    print(len(tstY_hat))
    print(prediY.shape)

    tstY_hat = np.vstack(tstY_hat)
    print(tstY_hat.shape)

    if torch.cuda.is_available():
        prediY = prediY.cpu().numpy()

    dim_y = 3
    with open(args.save_dir + "prop_pred.txt", "w") as e:
        for j in range(dim_y):
            #print([j, mean_absolute_error(tstY[:,j], prediY[:,j])], file=e)
            print("Dimension is: ", j, file=e)
            absolute_error = np.abs(tstY[:,j]- tstY_hat[:,j])
            print("mean: ", np.mean(absolute_error), file=e)
            print("std: ", np.std(absolute_error), file=e)
            print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])], file=e)


    with open(args.save_dir + "scaled_prop_pred.txt", "w") as e:
        for j in range(dim_y):
            #print([j, mean_absolute_error(tstY[:,j], prediY[:,j])], file=e)
            print("Dimension is: ", j, file=e)
            absolute_error = np.abs(scaled_tstY[:,j]- prediY[:,j])
            print("mean: ", np.mean(absolute_error), file=e)
            print("std: ", np.std(absolute_error), file=e)
            print([j, mean_absolute_error(scaled_tstY[:,j], prediY[:,j])], file=e)



    ## property targeting
    print('::: logP property optimisation')
    yid = 1
    ytarget = 4.8
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    maxlogp=[]

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "logp_prop_opt.txt", "w") as g:
        t = 0
        while t < 1000:
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[1] >= ytarget:
                print([t, smi, get_property(smi)], file=g)

                ytarget += 0.1
                ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
                t = 0
                maxlogp += [get_property(smi)[1]]

            t += 1
        maxlogp.sort()
        print(maxlogp, file=g)



    print('::: qed property optimisation')

    yid = 2
    ytarget = 0.93
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    maxqed = []

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "qed_prop_opt.txt", "w") as g:
        t = 0
        while t < 1000:
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[2] >= ytarget:
                print([t, smi, get_property(smi)], file=g)

                ytarget += 0.001
                ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
                t = 0
                maxqed += [get_property(smi)[2]]

            t += 1
        maxqed.sort()
        print(maxqed, file=g)


    print('::: logp property targeting')
    yid = 1
    ytarget = -2.25
    ymin = -2.5
    ymax = -2
    count = 0
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "logp-2.25_prop_targeting.txt", "w") as g:
        for t in range(1000):
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[1] >= ymin and get_property(smi)[1] <= ymax:
                print([t, smi, get_property(smi)], file=g)
                count +=1
        print("Count is: ", count, file=g)

    #−2.5 ≤ logP ≤ −2 5 ≤ logP ≤ 5.5 150 ≤ MW ≤ 200 500 ≤ MW ≤ 550

    print('::: logp property targeting')
    yid = 1
    ytarget = 5.25
    ymin = 5
    ymax = 5.5
    count = 0
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "logp5.25_prop_targeting.txt", "w") as g:
        for t in range(1000):
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[1] >= ymin and get_property(smi)[1] <= ymax:
                print([t, smi, get_property(smi)], file=g)
                count +=1
        print("Count is: ", count, file=g)


    print('::: molwt property targeting')
    yid = 0
    ytarget = 175
    ymin = 150
    ymax = 200
    count = 0
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "molwt175_prop_targeting.txt", "w") as g:
        for t in range(1000):
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[0] >= ymin and get_property(smi)[0] <= ymax:
                print([t, smi, get_property(smi)], file=g)
                count +=1
        print("Count is: ", count, file=g)


    print('::: molwt property targeting')
    yid = 0
    ytarget = 525
    ymin = 500
    ymax = 550
    count = 0
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "molwt525_prop_targeting.txt", "w") as g:
        for t in range(1000):
            smi = model.sampling_conditional(yid, ytarget_transform)
            #print([t, smi, get_property(smi)], file=g) # MolWt, LogP, QED
            if get_property(smi)[0] >= ymin and get_property(smi)[0] <= ymax:
                print([t, smi, get_property(smi)], file=g)
                count +=1
        print("Count is: ", count, file=g)



    ## unconditional generation
    print('::: unconditional generation')
    count_invalid = 0
    logp = []
    qed = []
    molwt = []
    with open(args.save_dir + "uncond_gen.txt", "w") as f:
        for t in range(1000):
            smi = model.sampling_unconditional()
            print([t, smi, get_property(smi)], file=f)
            if get_property(smi) == 'invalid':
                count_invalid +=1
            else:
                molwt += [get_property(smi)[0]]
                logp += [get_property(smi)[1]]
                qed += [get_property(smi)[2]]
        print("Molwt ", molwt, file=f)
        print("Logp ", logp, file=f)
        print("QED ", qed, file=f)
        print("count_invalid ", count_invalid, file=f)


    """
    ## conditional generation (e.g. MolWt=250)
    print('::: conditional generation (e.g. MolWt=250)')
    yid = 0
    ytarget = 250.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_molwt250.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)

    print('::: conditional generation (e.g. MolWt=350)')
    yid = 0
    ytarget = 350.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_molwt350.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)

    print('::: conditional generation (e.g. MolWt=450)')
    yid = 0
    ytarget = 450.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_molwt450.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)


    print('::: conditional generation (e.g. logP=5.3)')
    yid = 1
    ytarget = 5.3
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_logp53.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)

    print('::: conditional generation (e.g. logP=1.5)')
    yid = 1
    ytarget = 1.5
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_logp15.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)

    print('::: conditional generation (e.g. logP=3)')
    yid = 1
    ytarget = 3.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_logp3.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)


    print('::: conditional generation (e.g. logP=4.5)')
    yid = 1
    ytarget = 4.5
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_logp45.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)


    print('::: conditional generation (e.g. qed=0.5)')
    yid = 2
    ytarget = 0.5
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_qed0.5.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)


    print('::: conditional generation (e.g. qed=0.7)')
    yid = 2
    ytarget = 0.7
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_qed0.7.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)


    print('::: conditional generation (e.g. qed=0.9)')
    yid = 2
    ytarget = 0.9
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])

    print("ytarget_transform ", ytarget_transform)
    with open(args.save_dir + "cond_gen_qed0.9.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget_transform)
            print([t, smi, get_property(smi)], file=g)

    """
