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

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)

    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree


def proptensorize(data_pair, assm=True):
    smiles, props = data_pair
    mol_tree = PropMolTree(smiles, props)

    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=5)
    parser.add_option("--dataset", dest="dataset_name", default="zinc310k")

    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    print("Before Pool")
    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    print("Before open train data")

    if opts.dataset_name == "zinc310k":
        print("Dataset is ", opts.dataset_name)

        path = data_uri + opts.dataset_name + "/frac" + str(frac) + "/"
        with open(path + "train.pickle", "rb") as f:
            data = pickle.load(f)

        trnX_L = list(data['trnX_L'])
        trnY_L = list(data['trnY_L'])
        trnX_U = list(data['trnX_U'])
        #smiles = pd.read_csv(path + "ZINC_310k.csv").values[:ntrn+ntst,0] #0: SMILES
        #props = np.asarray(pd.read_csv(path + "ZINC_310k.csv").values[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED

        #smiles= trnX_L[:150]
        #props = trnY_L[:150]
        #trnX_U = trnX_U[:150]

        smiles= trnX_L
        props = trnY_L
        trnX_U = trnX_U

        data_pairs = []
        for id, smile in enumerate(smiles):
            #print(smile)
            data_pairs += [(smile, props[id])]

        print("Before pool map")
        all_data_prop = pool.map(proptensorize, data_pairs)
        all_data_u = pool.map(tensorize, trnX_U)

        le = int((len(all_data_prop) + num_splits - 1) / num_splits)

        print("Before write each tensor")
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data_prop[st:st + le]
            with open(opts.dataset_name +  "-processed/frac" + str(frac) + "/" + 'prop-tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f)

        le = int((len(all_data_u) + num_splits - 1) / num_splits)

        print("Before write each tensor")
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data_u[st:st + le]
            with open(opts.dataset_name + '-processed/frac' + str(frac) + "/" + 'u-tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f)


    else:
        with open(opts.train_path, "r") as f:
            data = [line.strip("\r\n ").split()[0] for line in f]
        print("Before pool map")
        all_data = pool.map(tensorize, data)
        le = int((len(all_data) + num_splits - 1) / num_splits)

        print("Before write each tensor")
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st:st + le]
            with open(opts.dataset_name + '-processed/tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f)
