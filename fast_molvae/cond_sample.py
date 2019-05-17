import torch
import torch.nn as nn
import pickle
import numpy
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fast_molvae_constants import *
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
    parser.add_argument('--testX', required=True) # ../data/zinc310k/test
    parser.add_argument('--testY', required=True) # ../data/zinc310k
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--save_dir', required=True)

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

    parser.add_argument('--infomax_true', type=int, default=1)
    parser.add_argument('--infomax_false', type=int, default=1)
    parser.add_argument('--u_kld_y', type=int, default=1)
    parser.add_argument('--ymse_factor', type=int, default=1)

    """
    print(get_property('Cc1ccc(CN2CC3CN(C)CCC3(C(=O)O)C2)o1'))
    print(get_property('Cc1ccc(S(=O)(=O)NCC(=O)Nc2ccc(C(C)O)cc2)cc1'))
    print(get_property('C=C(OCC)c1nc2ccccc2n1CCc1ccccc1'))
    print(get_property('CCOC(=O)c1ccc(NCC2CC2)c(F)c1'))
    print(get_property('CCOC(=O)c1nc(C)n(Cc2ccc(Cl)cc2)n1'))
    print(get_property('CC(=O)Nc1ccc(C(=O)n2nc(-c3ccco3)cc2CO)cc1'))
    print(get_property('COc1ccc(S(=O)(=O)NCc2ccc(C)cc2)cc1C(=O)NCC1CCCO1'))
    print(get_property('CN(C)C(=O)c1ccc2ncnc(NC(Cc3ccccc3O)C3CC3)c2c1'))
    print(get_property('COc1cc(C(=O)Nc2ccc(CN3CCOCC3)c(Cl)c2)nc(OC)n1'))
    print(get_property('CCC(O)CN1C(=O)COc2ccc(NC(=O)Cc3ccc4c(c3)OCO4)cc21'))

    print(get_property('COc1ccc(NC(=O)C2CCCNCC2)cc1OC'))
    print(get_property('CC(=O)NCc1ccc(C(=O)N(C)C2CCCC2)cc1'))
    print(get_property('CC(CC1CC1)NS(=O)(=O)c1ccc(F)cc1'))
    print(get_property('COC(=O)c1ccc(NC(C)CC2CC2)c(F)c1'))
    print(get_property('CCN(CC)C(=O)C1=CC2CC1c1cccnc12'))
    print(get_property('Cc1cc(C(=O)NCC2CCCc3ccccc32)on1'))
    print(get_property('COC(C)(Cc1ccccc1)c1nnnn1C1CCCC1'))
    print(get_property('CNc1nc(COC)ncc1C(=O)Nc1ccc(C)cc1'))
    print(get_property('OCCC(NC1CCc2ccccc21)C1CC1'))
    print(get_property('O=C(O)CN1CCC2(CC1c1ccccc1)OCCO2'))
    exit(0)
    """

    args = parser.parse_args()

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    tstY = torch.tensor(pickle.load(open(args.testY + "processed-testY.pkl", "rb")))

    """
    f = open(args.testX + "processed-test.pkl", "rb")
    data = pickle.load(f)
    tstX = list(data['tstX'])
    tstY = list(data['tstY'])

    with open(args.testX + "processed-testX.pkl", "wb") as g:
        pickle.dump(tstX, g)

    with open(args.testY + "processed-testY.pkl", "wb") as h:
        pickle.dump(tstY, h)
    print("tstX ", tstX[0])
    """

    """
    tstX = [tensorize(smile) for smile in tstX]
    testdict = {}
    testdict['tstX'] = tstX
    testdict['tstY'] = tstY
    with open(args.test + "processed-test.pkl", "wb") as g:
        pickle.dump(testdict, g)
    """

    model = CondJTNNVAE(vocab, args.hidden_size, args.prop_hidden_size, args.latent_size, args.depthT, args.depthG, args.infomax_true, args.infomax_false, args.u_kld_y, args.ymse_factor)

    state = torch.load(args.model) # need to change
    model.load_state_dict(state['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    loader = MolTreeFolder(args.testX, vocab, args.batch_size, num_workers=4)

    # model testing
    print('::: model testing')

    # property prediction performance
    print('::: property prediction performance')
    #tstX =  torch.tensor(tstX, dtype=torch.float32, device=cuda_device)
    #tstY_hat = []
    """
    prediY = []
    #it = iter(loader)
    nb = 0
    model.eval()
    for batch in loader:
        #if nb > 3:
        #    break
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = batch
        predi_y_L_mu, predi_y_L_lsgms = model.predi(*x_jtenc_holder)
        #tstY_hat_val = scaler_Y.inverse_transform(predi_y_L_mu.cpu().detach().numpy())
        #print("tstY_hat_val ", tstY_hat_val)
        #print("prediY", predi_y_L_mu)
        #tstY_hat += [tstY_hat_val]
        prediY += [predi_y_L_mu.detach()]

        nb +=1

    prediY = torch.cat(prediY)
    print(prediY.shape)
    if torch.cuda.is_available():
        prediY = prediY.cpu().numpy()

    dim_y = 3
    with open(args.save_dir + "prop_pred.txt", "w") as e:
        for j in range(dim_y):
            print([j, mean_absolute_error(tstY[:,j], prediY[:,j])], file=e)
    """

    ## unconditional generation
    print('::: unconditional generation')
    with open(args.save_dir + "uncond_gen.txt", "w") as f:
        for t in range(10):
            smi = model.sampling_unconditional()
            print([t, smi, get_property(smi)], file=f)

    ## conditional generation (e.g. MolWt=250)
    print('::: conditional generation (e.g. MolWt=250)')
    yid = 0
    ytarget = 250.
    #ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    with open(args.save_dir + "cond_gen_molwt250.txt", "w") as g:
        for t in range(10):
            smi = model.sampling_conditional(yid, ytarget)
            print([t, smi, get_property(smi)], file=g)
