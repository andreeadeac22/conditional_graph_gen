import pandas as pd
import numpy as np
import pickle
import rdkit

from fast_jtnn import *
from fast_molvae_constants import *

def split_data(dataset_name):
    file_name = data_uri + dataset_name +  "/" + 'ZINC_310k.csv'
    X = pd.read_csv(file_name).values[:ntrn+ntst,0] #0: SMILES
    Y = np.asarray(pd.read_csv(file_name).values[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED

    trndict = {}
    valdict = {}
    testdict = {}

    tstX=X[-ntst:]
    tstY=Y[-ntst:]

    X=X[:ntrn]
    Y=Y[:ntrn]

    nL=int(len(Y)*frac)
    nU=len(Y)-nL
    nL_trn=int(nL*(1-frac_val))
    nL_val=nL-nL_trn
    nU_trn=int(nU*(1-frac_val))
    nU_val=nU-nU_trn
    perm_id=np.random.permutation(len(Y))

    trnX_L=X[perm_id[:nL_trn]]
    trnY_L=Y[perm_id[:nL_trn]]

    valX_L=X[perm_id[nL_trn:nL_trn+nL_val]]
    valY_L=Y[perm_id[nL_trn:nL_trn+nL_val]]

    trnX_U=X[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]

    valX_U=X[perm_id[nL_trn+nL_val+nU_trn:]]

    trndict['trnX_L'] = trnX_L
    trndict['trnY_L'] = trnY_L
    trndict['trnX_U'] = trnX_U

    valdict['valX_L'] = valX_L
    valdict['valY_L'] = valY_L
    valdict['valX_U'] = valX_U

    testdict['tstX'] = tstX
    testdict['tstY'] = tstY

    with open(data_uri +  dataset_name + "/" + 'train.pickle', 'wb') as f:
        pickle.dump(trndict, f)
    with open(data_uri + dataset_name +  "/" + 'valid.pickle', 'wb') as f:
        pickle.dump(valdict, f)
    with open(data_uri + dataset_name +  "/" + 'test.pickle', 'wb') as f:
        pickle.dump(testdict, f)

def build_vocab(dataset_name):
        import sys
        cset = set()

        path = data_uri + dataset_name + "/"
        with open(path + "train.pickle", "rb") as f:
            data = pickle.load(f)

        trnX_L = list(data['trnX_L'])
        trnX_U = list(data['trnX_U'])


        print("trnX_L ", len(trnX_L))
        print("trnX_U ", len(trnX_U))

        X = list(trnX_L) + list(trnX_U)

        print(len(X))

        for smiles in X:
            #print("smiles ", smiles)
            mol = MolTree(smiles)
            for c in mol.nodes:
                cset.add(c.smiles)

        with open(data_uri + dataset_name + "/" + "vocab.txt", "w") as f:
            for x in cset:
                print(x, file=f)

#build_vocab("zinc310k")
