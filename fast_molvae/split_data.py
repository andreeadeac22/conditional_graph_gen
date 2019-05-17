import pandas as pd
import numpy as np
import pickle
import rdkit
from sklearn.preprocessing import StandardScaler

from fast_jtnn import *

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

    perm_id=np.random.permutation(len(Y))

    trnX_L=X[perm_id[:nL_trn]]
    trnY_L=Y[perm_id[:nL_trn]]

    valX_L=X[perm_id[nL_trn:nL_trn+nL_val]]
    valY_L=Y[perm_id[nL_trn:nL_trn+nL_val]]

    trnX_U=X[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]
    valX_U=X[perm_id[nL_trn+nL_val+nU_trn:]]

    scaler_Y = StandardScaler()
    scaler_Y.fit(Y)
    trnY_L=scaler_Y.transform(trnY_L)
    valY_L=scaler_Y.transform(valY_L)

    trndict['trnX_L'] = trnX_L
    trndict['trnY_L'] = trnY_L
    trndict['trnX_U'] = trnX_U

    valdict['valX_L'] = valX_L
    valdict['valY_L'] = valY_L
    valdict['valX_U'] = valX_U

    testdict['tstX'] = tstX
    testdict['tstY'] = tstY

    path = data_uri +  dataset_name + "/frac" + str(frac) + "/"

    with open(path + "stat_file.txt", "w") as stat_file:
        print("Fraction of labeled samples in training and validation sets (frac): ", frac, file=stat_file)
        print("Fraction of validation samples from whole non-test set (300k) (frac_val): ", frac_val, file=stat_file)
        print("trnX_L ", trnX_L.shape, file=stat_file)
        print("trnY_L ", trnY_L.shape, file=stat_file)
        print("trnX_U ", trnX_U.shape, file=stat_file)

        print("valX_L ", valX_L.shape, file=stat_file)
        print("valY_L ", valY_L.shape, file=stat_file)
        print("valX_U ", valX_U.shape, file=stat_file)

        print("tstX ", tstX.shape, file=stat_file)
        print("tstY ", tstY.shape, file=stat_file)


    with open(path + 'train.pickle', 'wb') as f:
        pickle.dump(trndict, f)
    with open(path + 'valid.pickle', 'wb') as f:
        pickle.dump(valdict, f)
    with open(path + 'test.pickle', 'wb') as f:
        pickle.dump(testdict, f)

def build_vocab(dataset_name):
        import sys
        cset = set()

        path = data_uri + dataset_name + "/frac" + str(frac) + "/"
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

        with open(path + "vocab.txt", "w") as f:
            for x in cset:
                print(x, file=f)

#build_vocab("zinc310k")
#split_data("zinc310k")
