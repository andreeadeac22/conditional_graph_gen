import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import _pickle as pickle
import rdkit
import gc
#import nvgpu

from fast_jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

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

parser.add_argument('--infomax', type=int, default=1)
parser.add_argument('--u_kld', type=int, default=1)

args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

if args.train == "zinc310k-processed":
    model = CondJTNNVAE(vocab, args.hidden_size, args.prop_hidden_size, args.latent_size, args.depthT, args.depthG, args.infomax, args.u_kld)
else:
    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
if torch.cuda.is_available():
    model = model.cuda()
print(model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

st_epoch = 0

if args.load_epoch > 0:
    checkpoint = torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    st_epoch = checkpoint['epoch']

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
meters = np.zeros(6)
prop_meters = np.zeros(6)
u_meters = np.zeros(6)

mem_file = open(args.save_dir + "track_mem.txt", "w")

if args.train == "zinc310k-processed":
    loader = SSMolTreeFolder(args.train, vocab, args.batch_size, num_workers=4) #make siamese dataloader
else:
    loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=4)

for epoch in range(st_epoch, args.epoch):
    print("Epoch ", epoch)
    #print("", file=mem_file, flush=True)
    #print("Epoch ", epoch, file=mem_file, flush=True)
    #print("", file=mem_file, flush=True)
    nb_tensors = 0
    nb_par = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if "Parameter" in str(type(obj)):
                    nb_par +=1
                else:
                    nb_tensors +=1
                #print(type(obj), obj.size(), file=mem_file, flush=True)
        except:
            pass
    print("nb_tensors ", nb_tensors)
    print("nb_par ", nb_par)
    #nbb = 0
    for batch in loader:
        torch.cuda.empty_cache()
        #if nbb > 5:
        #    break
        #nbb += 1
        #print("batch ")
        #print(batch)
        total_step += 1
        #try:
        model.zero_grad()
        loss, \
        prop_loss, prop_tree_kl, prop_mol_kl, prop_word_acc, prop_topo_acc, prop_assm_acc, prop_log_prior_y, \
        u_loss, tree_kl, mol_kl, u_word_acc, u_topo_acc, u_assm_acc,u_kld_y, \
        objYpred_MSE, d_true_loss, d_fake_loss = model(batch, beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        #except Exception as e:
        #    print(e)
        #    continue

        #meters = meters + np.array([tree_kl, mol_kl, wacc * 100, tacc * 100, sacc * 100])
        meters = meters + np.array([float(loss), float(prop_loss), float(u_loss), float(objYpred_MSE), float(d_true_loss), float(d_fake_loss)])
        prop_meters = prop_meters + np.array([float(prop_tree_kl), float(prop_mol_kl), float(prop_word_acc *100), float(prop_topo_acc*100), float(prop_assm_acc*100), float(prop_log_prior_y)])
        u_meters = u_meters + np.array([float(tree_kl), float(mol_kl), float(u_word_acc*100), float(u_topo_acc*100), float(u_assm_acc*100), float(u_kld_y)])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            prop_meters /= args.print_iter
            u_meters /= args.print_iter

            print("[%d] Beta: %.3f, Loss: %.2f, Prop_loss: %.2f, U_loss: %.2f, YMSE: %.2f, DTrue: %.2f, DFalse: %.2f, PNorm: %.2f, GNorm: %.2f" \
                %(total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))

            print("[%d] Prop_KL: %.2f, %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, Log_prior_y: %.2f" \
                % (total_step, prop_meters[0], prop_meters[1], prop_meters[2], prop_meters[3], prop_meters[4], prop_meters[5]))

            print("[%d] Unsup_KL: %.2f, %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, U_kld_y: %.2f" \
                % (total_step, u_meters[0], u_meters[1], u_meters[2], u_meters[3], u_meters[4], u_meters[5]))

            #print("Memory usage ", nvgpu.gpu_info(), file=mem_file)
            sys.stdout.flush()
            meters *= 0
            prop_meters *= 0
            u_meters *= 0

        if total_step % args.save_iter == 0:
            state = {'epoch': epoch, 'total_step': total_step, 'state_dict': model.state_dict(), \
                'optimizer': optimizer.state_dict()}
            torch.save(state,  args.save_dir + "/model.iter-" + str(total_step))
            #torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)
