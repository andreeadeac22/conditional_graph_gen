from __future__ import print_function

import torch
import numpy as np
import pandas as pd
from preprocessing import smiles_to_seq, vectorize, get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F

from torch_SSVAE import *
from util import *
from torch_constants import *

def run():
    # data preparation
    print('::: data preparation')

    smiles = pd.read_csv(data_uri).values[:ntrn+ntst,0] #0: SMILES
    Y = np.asarray(pd.read_csv(data_uri).values[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED

    list_seq = smiles_to_seq(smiles, char_set)
    Xs, X=vectorize(list_seq, char_set)


    tstX=X[-ntst:]
    tstXs=Xs[-ntst:]
    tstY=Y[-ntst:]

    X=X[:ntrn]
    Xs=Xs[:ntrn]
    Y=Y[:ntrn]

    nL=int(len(Y)*frac)
    nU=len(Y)-nL
    nL_trn=int(nL*(1-frac_val))
    nL_val=nL-nL_trn
    nU_trn=int(nU*(1-frac_val))
    nU_val=nU-nU_trn
    perm_id=np.random.permutation(len(Y))

    trnX_L=X[perm_id[:nL_trn]]
    trnXs_L=Xs[perm_id[:nL_trn]]
    trnY_L=Y[perm_id[:nL_trn]]

    valX_L=X[perm_id[nL_trn:nL_trn+nL_val]]
    valXs_L=Xs[perm_id[nL_trn:nL_trn+nL_val]]
    valY_L=Y[perm_id[nL_trn:nL_trn+nL_val]]

    trnX_U=X[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]
    trnXs_U=Xs[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]

    valX_U=X[perm_id[nL_trn+nL_val+nU_trn:]]
    valXs_U=Xs[perm_id[nL_trn+nL_val+nU_trn:]]

    scaler_Y = StandardScaler()
    scaler_Y.fit(Y)
    trnY_L=scaler_Y.transform(trnY_L)
    valY_L=scaler_Y.transform(valY_L)

    #trnX_L (142500, 89, 36)
    #trnXs_L (142500, 89, 36)
    #trnY_L (142500, 3)
    #valX_L (7500, 89, 36)
    #valXs_L (7500, 89, 36)
    #valY_L (7500, 3)

    #trnX_U (142500, 89, 36)
    #trnXs_U (142500, 89, 36)
    #valX_U (7500, 89, 36)
    #valXs_U (7500, 89, 36)

    ## model training
    print('::: model training')

    seqlen_x = X.shape[1]
    dim_x = X.shape[2]
    dim_y = Y.shape[1]

    batch_size_L=int(batch_size*len(trnX_L)/(len(trnX_L)+len(trnX_U)))
    batch_size_U=int(batch_size*len(trnX_U)/(len(trnX_L)+len(trnX_U)))
    n_batch=int(len(trnX_L)/batch_size_L)

    batch_size_val_L=int(len(valX_L)/10)
    batch_size_val_U=int(len(valX_U)/10)

    mu_prior=torch.tensor(np.mean(trnY_L,0), dtype=torch.float32)
    cov_prior=torch.tensor(np.cov(trnY_L.T), dtype=torch.float32)

    model = TorchSSVAE(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h, n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set)
    optimizer = torch.optim.Adam(model.parameters())

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(20):
        model.train()
        [trnX_L, trnXs_L, trnY_L]= permutation([trnX_L, trnXs_L, trnY_L])
        [trnX_U, trnXs_U]= permutation([trnX_U, trnXs_U])


        for i in range(n_batch):
            start_L=i*batch_size_L
            end_L=start_L+batch_size_L

            start_U=i*batch_size_U
            end_U=start_U+batch_size_U

            x_L =  torch.tensor(trnX_L[start_L:end_L], dtype=torch.float32)
            xs_L = torch.tensor(trnXs_L[start_L:end_L], dtype=torch.float32)
            y_L = torch.tensor(trnY_L[start_L:end_L], dtype=torch.float32)
            x_U =  torch.tensor(trnX_U[start_U:end_U], dtype=torch.float32)
            xs_U =  torch.tensor(trnXs_U[start_U:end_U], dtype=torch.float32)

            if torch.cuda.is_available():
                x_L = x_L.cuda()
                xs_L = xs_L.cuda()
                y_L = y_L.cuda()
                x_U = x_U.cuda()
                xs_U = xs_U.cuda()


            predictions = model(x_L, xs_L, y_L, x_U, xs_U)

            # complete loss, equation 3
            #loss = (train_objL * float(batch_size_L) + train_objU * float(batch_size_U))/float(batch_size_L+batch_size_U) + float(batch_size_L)/float(batch_size_L+batch_size_U) * (beta * train_objYpred_MSE)
            criterion =  nn.MSELoss()
            loss = criterion(predictions, y_L)

            model.zero_grad()
            #start = time.time()
            loss.backward()
            optimizer.step()
            #total_time += time.time() - start

            #print("loss", loss)


        model.eval()
        val_res = []
        for i in range(10):
            start_L=i*batch_size_val_L
            end_L=start_L+batch_size_val_L

            start_U=i*batch_size_val_U
            end_U=start_U+batch_size_val_U

            #val_cost, val_objL, val_objU, val_objYpred_MSE = \'
            x_L =  torch.tensor(valX_L[start_L:end_L], dtype=torch.float32)
            xs_L = torch.tensor(valXs_L[start_L:end_L], dtype=torch.float32)
            y_L = torch.tensor(valY_L[start_L:end_L], dtype=torch.float32)
            x_U =  torch.tensor(valX_U[start_U:end_U], dtype=torch.float32)
            xs_U =  torch.tensor(valXs_U[start_U:end_U], dtype=torch.float32)

            if torch.cuda.is_available():
                x_L = x_L.cuda()
                xs_L = xs_L.cuda()
                y_L = y_L.cuda()
                x_U = x_U.cuda()
                xs_U = xs_U.cuda()

            val_predictions = model(x_L, xs_L, y_L, x_U, xs_U)
            # val_res.append([val_cost, val_objL, val_objU, val_objYpred_MSE])
            val_loss = criterion(val_predictions, y_L)
            print("val_loss", loss)


            """
            val_res.append([val_loss])

            val_res=np.mean(val_res,axis=0)
            print(epoch, ['Training', 'cost_trn', trn_res[1]])
            print('---', ['Validation', 'cost_val', val_res[0]])

            val_log[epoch] = val_res[0]
            if epoch > 20 and np.min(val_log[0:epoch-10]) * 0.99 < np.min(val_log[epoch-10:epoch+1]):
                print('---termination condition is met')
                break
            """


if __name__ == '__main__':
    run()
