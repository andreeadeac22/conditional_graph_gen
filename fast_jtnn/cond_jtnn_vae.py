import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit
import rdkit.Chem as Chem
import copy, math

from .mol_tree import Vocab, MolTree
from .nnutils import create_var, flatten_tensor, avg_pool
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .jtnn_predi import JTNNPredi
from .mpn import MPN
from .jtmpn import JTMPN
from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
from .constants import *


class CondJTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, prop_hidden_size, latent_size, depthT, depthG):
        super(CondJTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = int(latent_size / 2) #Tree and Mol has two vectors

        self.predi = JTNNPredi(hidden_size, prop_hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.decoder = JTNNDecoder(vocab, hidden_size, prop_hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        self.prop_dense = nn.Linear(num_prop, prop_hidden_size)
        self.relu = nn.ReLU()

        self.tree_enc_dense = nn.Linear(hidden_size + prop_hidden_size, hidden_size)
        self.mol_enc_dense = nn.Linear(hidden_size + prop_hidden_size, hidden_size)

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size + prop_hidden_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

    def encode(self, jtenc_holder, mpn_holder, props):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)

        prop_tree_vecs = self.tree_enc_dense(torch.cat((tree_vecs, props), dim=1))
        prop_mol_vecs = self.mol_enc_dense(torch.cat((mol_vecs, props), dim=1))

        return prop_tree_vecs, tree_mess, prop_mol_vecs


    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def draw_sample(self, mu, lsgms):
        epsilon = torch.randn(mu.shape).to(device) #by default, mu=0, std=1
        exp_lsgms = torch.exp(0.5*lsgms).to(device)
        sample = torch.add(mu, (exp_lsgms*epsilon))
        return sample

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size)
        z_mol = torch.randn(1, self.latent_size)
        if torch.cuda.is_available():
            z_tree = z_tree.cuda()
            z_mol = z_mol.cuda()
        return self.decode(z_tree, z_mol, prob_decode)


    def forward(self, x_batch, beta):
        print("Conditional")
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder, \
            prop_x_batch, prop_x_jtenc_holder, prop_x_mpn_holder, prop_x_jtmpn_holder, props \
                = x_batch
        props = self.relu(self.prop_dense(props))
        #Labeled
        y_L_mu, y_L_lsgms = self.predi(*prop_x_jtenc_holder)
        prop_x_tree_vecs, prop_x_tree_mess, prop_x_mol_vecs = self.encode(prop_x_jtenc_holder, prop_x_mpn_holder, props)
        prop_z_tree_vecs, prop_tree_kl = self.rsample(prop_x_tree_vecs, self.T_mean, self.T_var)
        prop_z_mol_vecs, prop_mol_kl = self.rsample(prop_x_mol_vecs, self.G_mean, self.G_var)

        cat_decoder = torch.cat((prop_z_tree_vecs, props), dim=1)
        print("prop_x_batch ", len(prop_x_batch))
        print("cat_decoder ", cat_decoder.shape)

        prop_word_loss, prop_topo_loss, prop_word_acc, prop_topo_acc = self.decoder(prop_x_batch, cat_decoder)

        print("prop_z_mol_vecs ", prop_z_mol_vecs.shape)
        print("props ", props.shape)
        print("prop_x_tree_mess ", prop_x_tree_mess.shape)

        cat_z_mol = torch.cat((prop_z_mol_vecs, props), dim=1)
        #cat_tree = torch.cat((prop_x_tree_mess, props), dim=1)
        prop_assm_loss, prop_assm_acc = self.assm(prop_x_batch, prop_x_jtmpn_holder, cat_z_mol, prop_x_tree_mess)

        prop_kl_div = prop_tree_kl + prop_mol_kl
        prop_loss = prop_word_loss + prop_topo_loss + prop_assm_loss + beta * prop_kl_div

        #Unlabeled
        y_U_mu, y_U_lsgms = self.predi(*x_jtenc_holder)
        y_U_sample = self.draw_sample(y_U_mu, y_U_lsgms)
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder, y_U_sample)
        print("x_tree_vecs ", x_tree_vecs.shape)
        print("x_tree_mess ", x_tree_mess.shape)
        print("x_mol_vecs ", x_mol_vecs.shape)

        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        u_cat_decoder = torch.cat((z_tree_vecs, y_U_sample), dim=1)
        u_word_loss, u_topo_loss, u_word_acc, u_topo_acc = self.decoder(x_batch, u_cat_decoder)

        u_cat_z_mol = torch.cat((z_mol_vecs, y_U_sample), dim=1)
        u_assm_loss, u_assm_acc = self.assm(x_batch, x_jtmpn_holder, u_cat_z_mol, x_tree_mess)

        u_kl_div = tree_kl + mol_kl
        u_loss = u_word_loss + u_topo_loss + u_assm_loss + beta * u_kl_div

        return prop_loss + u_loss, prop_tree_kl.item(), prop_mol_kl.item(), prop_word_acc, prop_topo_acc, prop_assm_acc
                #u_loss, tree_kl.item(), mol_kl.item(), u_word_acc, u_topo_acc, u_assm_acc



    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt


    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode, props):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode, props)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None


    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None:
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol