from __future__ import print_function

import torch
import numpy as np
import pandas as pd
from preprocessing import smiles_to_seq, vectorize, get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        print("On cuda")
        model.cuda()
    else:
        print("Not on cuda")

    for epoch in range(20):
        model.train()
        [trnX_L, trnXs_L, trnY_L]= permutation([trnX_L, trnXs_L, trnY_L])
        [trnX_U, trnXs_U]= permutation([trnX_U, trnXs_U])


        for i in range(n_batch):
            start_L=i*batch_size_L
            end_L=start_L+batch_size_L

            start_U=i*batch_size_U
            end_U=start_U+batch_size_U

            x_L =  torch.tensor(trnX_L[start_L:end_L], dtype=torch.float32, device=device)
            xs_L = torch.tensor(trnXs_L[start_L:end_L], dtype=torch.float32, device=device)
            y_L = torch.tensor(trnY_L[start_L:end_L], dtype=torch.float32, device=device)
            x_U =  torch.tensor(trnX_U[start_U:end_U], dtype=torch.float32, device=device)
            xs_U =  torch.tensor(trnXs_U[start_U:end_U], dtype=torch.float32, device=device)

            objL_res, objU_res, objYpred, predictor_L_out = model(x_L, xs_L, y_L, x_U, xs_U)
            loss = (objL_res * float(batch_size_L) + objU_res * float(batch_size_U))/float(batch_size_L+batch_size_U) + float(batch_size_L)/float(batch_size_L+batch_size_U) * (beta * objYpred)

            eval_y_L = y_L.cpu().detach().numpy()
            eval_predictor_L_out = predictor_L_out.cpu().detach().numpy()
            mae = mean_absolute_error(eval_y_L, eval_predictor_L_out)
            rmse = np.sqrt(mean_squared_error(eval_y_L, eval_predictor_L_out))
            print("Epoch {0:2d} | Batch {1:4d} : MAE {2:2.3f}, RMSE {3:2.3f}, Loss {4:5.3f}".format(epoch, i, mae, rmse, loss.item()))

            model.zero_grad()
            #start = time.time()
            loss.backward()
            optimizer.step()
            #total_time += time.time() - start

        ## model validation
        print('::: model validation')

        model.eval()
        val_res = []
        for i in range(10):
            start_L=i*batch_size_val_L
            end_L=start_L+batch_size_val_L

            start_U=i*batch_size_val_U
            end_U=start_U+batch_size_val_U

            x_L =  torch.tensor(valX_L[start_L:end_L], dtype=torch.float32, device=device)
            xs_L = torch.tensor(valXs_L[start_L:end_L], dtype=torch.float32, device=device)
            y_L = torch.tensor(valY_L[start_L:end_L], dtype=torch.float32, device=device)
            x_U =  torch.tensor(valX_U[start_U:end_U], dtype=torch.float32, device=device)
            xs_U =  torch.tensor(valXs_U[start_U:end_U], dtype=torch.float32, device=device)

            objL_res, objU_res, objYpred, predictor_L_out_valid = model(x_L, xs_L, y_L, x_U, xs_U)
            # val_res.append([val_cost, val_objL, val_objU, val_objYpred_MSE])

            eval_y_L = y_L.cpu().detach().numpy()
            eval_predictor_L_out_valid = predictor_L_out_valid.cpu().detach().numpy()

            mae_valid= mean_absolute_error(eval_y_L, eval_predictor_L_out_valid)
            rmse_valid = np.sqrt(mean_squared_error(eval_y_L, eval_predictor_L_out_valid))
            print("Valid epoch {0:2d} : MAE{1:2.3f}, RMSE{1:2.3f} ".format(epoch, mae_valid, rmse_valid))

    # property prediction performance
    print('::: property prediction performance')
    predi_Y_mu, predi_Y_lsgms = model.rnn_predictor(tstX)
    tstY_hat = scaler_Y.inverse_transform(predi_Y_mu)
     for j in range(dim_y):
        print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])

    ## unconditional generation
    print('::: unconditional generation')
    for t in range(10):
        smi = model.sampling_unconditional()
        print([t, smi, get_property(smi)])

    





if __name__ == '__main__':
    run()
