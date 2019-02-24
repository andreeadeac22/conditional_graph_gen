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
        self.predi_dense = torch.nn.Linear(self.dim_h * 2, self.dim_y)

        self.enc_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers=self.n_hidden, batch_first=True, bidirectional=True)
        self.enc_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.enc_peek_dense = torch.nn.Linear(self.dim_y, self.dim_x)
        self.enc_dense = torch.nn.Linear(self.dim_h * 2, self.dim_z * 2)

        self.dec_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers = self.n_hidden, batch_first=True)
        self.dec_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.dec_peek_dense = torch.nn.Linear(self.dim_y, self.dim_x)
        self.dec_dense = torch.nn.Linear(self.dim_h, self.dim_y)



    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        print("y_L.shape", y_L.shape)
        mu_prior = y_L.mean()
        cov_prior = torch.transpose(y_L, 0, 1)

        encoder_L_out = self.rnn_encoder(x_L, y_L)
        print("encoder_L_out.shape", encoder_L_out.shape)
        z_L_mu, z_L_lsgms = torch.split(encoder_L_out, [self.dim_z, self.dim_z], dim=1)
        print("z_L_mu ", z_L_mu.shape)
        print("z_L_lsgms", z_L_lsgms.shape)
        z_L_sample = self.draw_sample(z_L_mu, z_L_lsgms) # 100,100
        decoder_L_out = self.rnn_decoder(xs_L, torch.cat([z_L_sample, y_L], dim=1))
        x_L_recon = F.softmax(decoder_L_out)

        objL_res = objL(x_L, x_L_recon, y_L, z_L_mu, z_L_lsgms, mu_prior, cov_prior)
        return objL_res


    def rnn_predictor(self, x_L):
        zero_state = torch.zeros(self.n_hidden *2, x_L.shape[0], self.dim_h).to(device)
        _, final_state = self.predi_gru(x_L, zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        predi_res = predi_dense(cat_fw_bw)
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
        print("xs_L", x.shape)
        print("z sample concatenated with y ", st.shape)

        zero_state = torch.zeros(self.n_hidden, x.shape[0], self.dim_y).to(device)
        print("zero state", zero_state.shape)

        zero_state = F.sigmoid(self.dec_zero_dense(zero_state))

        peek_in = F.sigmoid(self.dec_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        rnn_outputs, final_state = self.dec_gru(torch.cat([x, peek], dim=2), zero_state)
        dec_res = self.dec_dense(rnn_outputs)
        print("dec_res shape ", dec_res.shape)
        return dec_res


    def draw_sample(self, mu, lsgms):
        epsilon = torch.randn(mu.shape).to(device) #by default, mu=0, std=1
        exp_lsgms = torch.exp(0.5*lsgms).to(device)
        sample = torch.add(mu, (exp_lsgms*epsilon))
        return sample

    def objL(self, x_L, x_L_recon, y_L, z_L_mu, z_L_lsgms, cov_prior, tf_mu_prior, tf_cov_prior):
        L_log_lik = compute_log_lik(x_L, x_L_recon)
        L_log_prior_y = noniso_logpdf(y_L, cov_prior, tf_mu_prior, tf_cov_prior)
        L_KLD_z = iso_KLD(z_L_mu, z_L_lsgms)

        objL_res = - torch.reduce_mean(L_log_lik + L_log_prior_y - L_KLD_z)
        return objL_res
