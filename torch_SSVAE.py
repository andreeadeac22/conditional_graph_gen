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
        self.predi_dense = torch.nn.Linear(self.dim_h * 2, self.dim_y * 2)

        self.enc_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers=self.n_hidden, batch_first=True, bidirectional=True)
        self.enc_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.enc_peek_dense = torch.nn.Linear(self.dim_y, self.dim_x)
        self.enc_dense = torch.nn.Linear(self.dim_h * 2, self.dim_z * 2)

        self.dec_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers = self.n_hidden, batch_first=True)
        self.dec_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.dec_peek_dense = torch.nn.Linear(self.dim_y + self.dim_z, self.dim_x)
        self.dec_dense = torch.nn.Linear(self.dim_h, self.dim_x)




    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        #  y_L.shape is batch x dim_y (100,3) -- 3 features
        mu_prior = torch.unsqueeze(torch.mean(y_L, 0), dim=0) # (1,3)
        cov_prior = torch.Tensor(np.cov(torch.transpose(y_L, 0, 1)), device=device) #(3,3)

        # labeled computation
        encoder_L_out = self.rnn_encoder(x_L, y_L)         # encoder_L_out.shape is batch x 2*dim_z (100,200)
        z_L_mu, z_L_lsgms = torch.split(encoder_L_out, [self.dim_z, self.dim_z], dim=1)         #  z_L_mu.shape is batch x dim_z (100,100),  z_L_lsgms.shape is batch x dim_z (100,100)
        z_L_sample = self.draw_sample(z_L_mu, z_L_lsgms) # 100,100
        decoder_L_out = self.rnn_decoder(xs_L, torch.cat([z_L_sample, y_L], dim=1))
        x_L_recon = F.softmax(decoder_L_out)
        objL_res = self.objL(x_L, x_L_recon, y_L, z_L_mu, z_L_lsgms, mu_prior, cov_prior)

        #unlabeled computation
        predictor_U_out = self.rnn_predictor(x_U)
        y_U_mu, y_U_lsgms = torch.split(predictor_U_out, [self.dim_y, self.dim_y], 1)
        y_U_sample = self._draw_sample(y_U_mu, y_U_lsgms)
        encoder_U_out = self.rnn_encoder(x_U, y_U_sample)
        z_U_mu, z_U_lsgms = torch.split(encoder_U_out, [self.dim_z, self.dim_z], dim=1)
        z_U_sample = self.draw_sample(z_U_mu, z_U_lsgms) # 100,100
        decoder_U_out = self.rnn_decoder(xs_U, torch.cat([z_U_sample, y_U_sample], dim=1))
        x_U_recon = F.softmax(decoder_U_out)
        objU_res = self.objU(x_U, x_U_recon, y_U_mu, y_U_lsgms, z_U_mu, z_U_lsgms, mu_prior, cov_prior)

        cost = objL_res + objU_res + objYpred
        return cost


    def rnn_predictor(self, x_L):
        zero_state = torch.zeros(self.n_hidden *2, x_L.shape[0], self.dim_h).to(device)
        _, final_state = self.predi_gru(x_L, zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        predi_res = self.predi_dense(cat_fw_bw)
        return predi_res


    def rnn_encoder(self, x, st):
        zero_state = torch.zeros(self.n_hidden *2, x.shape[0], self.dim_y).to(device)
        zero_state = F.sigmoid(self.enc_zero_dense(zero_state))

        peek_in = F.sigmoid(self.enc_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        _, final_state = self.enc_gru(torch.cat([x, peek], dim=2), zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        enc_res = self.enc_dense(cat_fw_bw)
        return enc_res


    def rnn_decoder(self, x, st):
        # x.shape is batch x  seq_len x alphabet_size (100,89.36)
        # (z sample concatenated with y) st.shape is (100,103)
        zero_state = torch.zeros(self.n_hidden, x.shape[0], self.dim_y).to(device) # zero_state.shape is (3,100,3)
        zero_state = F.sigmoid(self.dec_zero_dense(zero_state))

        peek_in = F.sigmoid(self.dec_peek_dense(st))
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
