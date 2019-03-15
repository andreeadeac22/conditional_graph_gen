import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle
import rdkit

from fast_jtnn import *

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

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    print("Before Pool")
    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    print("Before open train data")
    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    print("data length", len(data))

    data = data[:150]


    print("Before pool map")
    all_data = pool.map(tensorize, data)

    le = int((len(all_data) + num_splits - 1) / num_splits)

    print("all_data", all_data[0].data)


    print("Before write each tensor")
    for split_id in range(num_splits):
        st = split_id * le

        sub_data = all_data[st:st + le]

        with open('processed/moses_tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
