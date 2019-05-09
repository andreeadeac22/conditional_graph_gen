import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle
import rdkit
import pandas as pd
import numpy as np

from fast_jtnn import *
from fast_molvae_constants import *

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("--dataset", dest="dataset_name", default="zinc310k")

    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    print("Before Pool")
    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    print("Before open train data")

    if opts.dataset_name == "zinc310k":
        print("Dataset is ", opts.dataset_name)

        path = data_uri + opts.dataset_name + "/"
        with open(path + "train.pickle", "rb") as f:
            data = pickle.load(f)

        trnX_L = list(data['trnX_L'])
        trnY_L = list(data['trnY_L'])
        trnX_U = list(data['trnX_U'])

        print(trnY_L[0])

        trnY_L = np.stack(trnY_L, axis=0)
        print(trnY_L.shape)

        mu_prior=np.mean(trnY_L,0)
        cov_prior=np.cov(trnY_L.T)

        print(mu_prior)

        print(cov_prior)

        print(cov_prior[0][1], cov_prior[0][2])
