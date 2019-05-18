import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import _pickle as pickle
import os, random, re

from .mol_tree import MolTree
from .jtnn_enc import JTNNEncoder
from .mpn import MPN
from .jtmpn import JTMPN


class SSMolTreeFolder(object):
    def __init__(self, data_folder, vocab, batch_size_L, batch_size_U, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]

        regex = re.compile(r'prop')
        regex_u = re.compile(r'u')

        self.data_files_prop = list(filter(regex.match, self.data_files))
        self.data_files_u = list(filter(regex_u.match, self.data_files))

        print("prop ", self.data_files_prop)
        print("u ", self.data_files_u)

        #self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.batch_size_L = batch_size_L
        self.batch_size_U = batch_size_U

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        print("SSMolTreeFolder iter")
        for id in range(len(self.data_files_u)):
            #if id > 3: continue
            fn_prop = os.path.join(self.data_folder,self.data_files_prop[id])
            fn_u = os.path.join(self.data_folder,self.data_files_u[id])
            print("fn ", fn_prop, fn_u)
            with open(fn_prop, "rb") as f:
                with open(fn_u, "rb") as g:
                    try:
                        print("Before load")
                        data_prop = pickle.load(f)
                        data_u = pickle.load(g)
                        print("Loaded")
                        if self.shuffle:
                            idx = np.random.permutation(len(data_prop))
                            data_prop = [data_prop[j] for j in idx]
                            data_u = [data_u[j] for j in idx]
                            #random.shuffle(data_prop) #shuffle data before batch
                        j= 0
                        batches = []
                        for i in range(0, len(data_u), self.batch_size_U):
                            batches += [(data_u[i : i + self.batch_size_U], data_prop[j : j + self.batch_size_L])]
                            j += self.batch_size_L

                        if len(batches[-1]) < self.batch_size_U:
                            batches.pop()

                        dataset = SSMolTreeDataset(batches, self.vocab, self.assm)
                        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

                        for b in dataloader:
                            yield b

                        del data_prop, data_u, batches, dataset, dataloader
                    except(UnicodeDecodeError) as e:
                        print("Error!")


class SSMolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)


def tensorize(pair_batch, vocab, assm=True):
    tree_batch, prop_tree_batch = pair_batch

    set_batch_nodeID(tree_batch, vocab)
    set_batch_nodeID(prop_tree_batch, vocab)

    smiles_batch = [tree.smiles for tree in tree_batch]
    prop_smiles_batch = [tree.smiles for tree in prop_tree_batch]

    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    prop_jtenc_holder, prop_mess_dict = JTNNEncoder.tensorize(prop_tree_batch)
    prop_jtenc_holder = prop_jtenc_holder
    prop_mpn_holder = MPN.tensorize(prop_smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder, \
            prop_tree_batch, prop_jtenc_holder, prop_mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)


    prop_cands = []
    prop_batch_idx = []
    props = []
    for i,mol_tree in enumerate(prop_tree_batch):
        props.append(mol_tree.props)
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            prop_cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            prop_batch_idx.extend([i] * len(node.cands))
    prop_jtmpn_holder = JTMPN.tensorize(prop_cands, prop_mess_dict)
    prop_batch_idx = torch.LongTensor(prop_batch_idx)
    props = torch.Tensor(props)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx), \
        prop_tree_batch, prop_jtenc_holder, prop_mpn_holder, (prop_jtmpn_holder, prop_batch_idx), props


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
