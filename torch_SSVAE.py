import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_constants import *
from util import *
from constants import *

class TorchSSVAE(nn.Module):
    def __init__(self, seqlen_x, dim_x, dim_y, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, beta=10000, char_set=[' ']):
        super(TorchSSVAE, self).__init__()
        self.seqlen_x, self.dim_x, self.dim_y, self.dim_z, self.dim_h, self.n_hidden, self.batch_size = seqlen_x, dim_x, dim_y, dim_z, dim_h, n_hidden, batch_size

        self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
        self.int_to_char = dict((i,c) for i,c in enumerate(char_set))

        self.predi_gru = torch.nn.GRU(self.dim_x, self.dim_h, num_layers=self.n_hidden, batch_first=True, bidirectional=True)
        self.predi_dense_mu = torch.nn.Linear(self.dim_h * 2, self.dim_y)
        self.predi_dense_lsgms = torch.nn.Linear(self.dim_h * 2, self.dim_y)


        self.enc_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers=self.n_hidden, batch_first=True, bidirectional=True)
        self.enc_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.enc_peek_dense = torch.nn.Linear(self.dim_y, self.dim_x)
        self.enc_dense = torch.nn.Linear(self.dim_h * 2, self.dim_z * 2)

        self.dec_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers = self.n_hidden, batch_first=True)
        self.dec_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.dec_peek_dense = torch.nn.Linear(self.dim_y + self.dim_z, self.dim_x)
        self.dec_dense = torch.nn.Linear(self.dim_h, self.dim_x)

        self.softmax = torch.nn.Softmax(dim=2)




    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        #  y_L.shape is batch x dim_y (100,3) -- 3 features
        mu_prior = torch.unsqueeze(torch.mean(y_L, 0), dim=0) # (1,3)
        y_L_transpose = torch.transpose(y_L, 0, 1)
        np_cov = np.cov(y_L_transpose.cpu())
        cov_prior = torch.Tensor(np_cov).to(device) #(3,3)

        # labeled computation
        y_L_mu, y_L_lsgms = self.rnn_predictor(x_L)
        encoder_L_out = self.rnn_encoder(x_L, y_L)         # encoder_L_out.shape is batch x 2*dim_z (100,200)
        z_L_mu, z_L_lsgms = torch.split(encoder_L_out, [self.dim_z, self.dim_z], dim=1)         #  z_L_mu.shape is batch x dim_z (100,100),  z_L_lsgms.shape is batch x dim_z (100,100)
        z_L_sample = self.draw_sample(z_L_mu, z_L_lsgms) # 100,100
        decoder_L_out = self.rnn_decoder(xs_L, torch.cat([z_L_sample, y_L], dim=1))
        x_L_recon = self.softmax(decoder_L_out)
        objL_res = self.objL(x_L, x_L_recon, y_L, z_L_mu, z_L_lsgms, mu_prior, cov_prior)

        #unlabeled computation
        y_U_mu, y_U_lsgms  = self.rnn_predictor(x_U)
        y_U_sample = self.draw_sample(y_U_mu, y_U_lsgms)
        encoder_U_out = self.rnn_encoder(x_U, y_U_sample)
        z_U_mu, z_U_lsgms = torch.split(encoder_U_out, [self.dim_z, self.dim_z], dim=1)
        z_U_sample = self.draw_sample(z_U_mu, z_U_lsgms) # 100,100
        decoder_U_out = self.rnn_decoder(xs_U, torch.cat([z_U_sample, y_U_sample], dim=1))
        x_U_recon = self.softmax(decoder_U_out)
        objU_res = self.objU(x_U, x_U_recon, y_U_mu, y_U_lsgms, z_U_mu, z_U_lsgms, mu_prior, cov_prior)
        objYpred_MSE = torch.mean(torch.sum((y_L-y_L_mu) * (y_L-y_L_mu), dim=1))

        return objL_res, objU_res, objYpred_MSE, y_U_mu


    def rnn_predictor(self, x_L):
        zero_state = torch.zeros(self.n_hidden *2, x_L.shape[0], self.dim_h).to(device)
        _, final_state = self.predi_gru(x_L, zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        predi_res_mu = self.predi_dense_mu(cat_fw_bw)
        predi_res_lsgms = self.predi_dense_lsgms(cat_fw_bw)
        return predi_res_mu, predi_res_lsgms


    def rnn_encoder(self, x, st):
        zero_state = torch.zeros(self.n_hidden *2, x.shape[0], self.dim_y).to(device)
        zero_state = torch.sigmoid(self.enc_zero_dense(zero_state))

        peek_in = torch.sigmoid(self.enc_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        _, final_state = self.enc_gru(torch.cat([x, peek], dim=2), zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        enc_res = self.enc_dense(cat_fw_bw)
        return enc_res


    def rnn_decoder(self, x, st):
        # x.shape is batch x  seq_len x alphabet_size (100,89.36)
        # (z sample concatenated with y) st.shape is (100,103)
        zero_state = torch.zeros(self.n_hidden, x.shape[0], self.dim_y).to(device) # zero_state.shape is (3,100,3)
        zero_state = torch.sigmoid(self.dec_zero_dense(zero_state))

        peek_in = torch.sigmoid(self.dec_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        rnn_outputs, final_state = self.dec_gru(torch.cat([x, peek], dim=2), zero_state)
        dec_res = self.dec_dense(rnn_outputs)
        # dec_res.shape is (100,89,3)
        return dec_res


    def draw_sample(self, mu, lsgms):
        epsilon = torch.randn(mu.shape).to(device) #by default, mu=0, std=1
        exp_lsgms = torch.exp(0.5*lsgms).to(device)
        sample = torch.add(mu, (exp_lsgms*epsilon))
        return sample


    def objL(self, x_L, x_L_recon, y_L, z_L_mu, z_L_lsgms, mu_prior, cov_prior):
        # x_L.shape is (100,89,36)
        # x_L_recon.shape is (100,89,36)
        L_log_lik = compute_log_lik(x_L, x_L_recon)
        L_log_prior_y = noniso_logpdf(y_L, mu_prior, cov_prior)
        L_KLD_z = iso_KLD(z_L_mu, z_L_lsgms)
        objL_res = - torch.mean(L_log_lik + L_log_prior_y - L_KLD_z)
        return objL_res


    def objU(self, x_U, x_U_recon, y_U_mu, y_U_lsgms, z_U_mu, z_U_lsgms, mu_prior, cov_prior):
        U_log_lik = compute_log_lik(x_U, x_U_recon)
        U_KLD_y = noniso_KLD(mu_prior, cov_prior, y_U_mu, y_U_lsgms)
        U_KLD_z = iso_KLD(z_U_mu, z_U_lsgms)
        objU_res = - torch.mean(U_log_lik - U_KLD_y - U_KLD_z)
        return objU_res


    def sampling_unconditional(self):
        sample_z=np.random.randn(1, self.dim_z)
        sample_y=np.random.multivariate_normal(self.mu_prior, self.cov_prior, 1)
        sample_smiles=self.beam_search(sample_z, sample_y, k=5)
        return sample_smiles


    def beam_search(self, z_input, y_input, k=5, x_G_recon, ):
        def reconstruct(xs_input, z_sample, y_input):
            decoder_G_out = self.rnn_decoder(xs_input, torch.cat([z_sample, y_input], dim=1))
            x_G_recon = self.softmax(decoder_G_out)
            return x_G_recon

        cands=np.asarray([np.zeros((1, self.seqlen_x, self.dim_x), dtype=np.float32)] )
        cands_score=np.asarray([100.])

        for i in range(self.seqlen_x-1):
            cands2=[]
            cands2_score=[]
            for j, samplevec in enumerate(cands):
                o = reconstruct(samplevec, z_input, y_input)
                sampleidxs = np.argsort(-o[0,i])[:k]
                for sampleidx in sampleidxs:
                    samplevectt=np.copy(samplevec)
                    samplevectt[0, i+1, sampleidx] = 1.
                    cands2.append(samplevectt)
                    cands2_score.append(cands_score[j] * o[0,i,sampleidx])

            cands2_score=np.asarray(cands2_score)
            cands2=np.asarray(cands2)
            kbestid = np.argsort(-cands2_score)[:k]
            cands=np.copy(cands2[kbestid])
            cands_score=np.copy(cands2_score[kbestid])
            if np.sum([np.argmax(c[0][i+1]) for c in cands])==0:
                break
        sampletxt = ''.join([self.int_to_char[np.argmax(t)] for t in cands[0,0]]).strip()
        return sampletxt
