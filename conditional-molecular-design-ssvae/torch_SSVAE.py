import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_constants import *

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
        self.enc_dense = torch.nn.Linear(self.dim_h * 2, self.dim_y)

        self.dec_gru = torch.nn.GRU(self.dim_x *2, self.dim_h, num_layers = self.n_hidden, batch_first=True)
        self.dec_zero_dense = torch.nn.Linear(self.dim_y , self.dim_h)
        self.dec_peek_dense = torch.nn.Linear(self.dim_y, self.dim_x)
        self.dec_dense = torch.nn.Linear(self.dim_h, self.dim_y)



    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        
        #predi_res = self.rnn_predictor(x_L)
        dec_res = self.rnn_decoder(x_L, y_L)
        return dec_res


    def rnn_predictor(self, x_L):
        zero_state = torch.zeros(self.n_hidden *2, x_L.shape[0], self.dim_h)
        if torch.cuda.is_available():
            zero_state = zero_state.cuda()
        _, final_state = self.predi_gru(x_L, zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        predi_res = predi_dense(cat_fw_bw)
        return predi_res


    def rnn_encoder(self, x, st):
        zero_state = torch.zeros(self.n_hidden *2, x.shape[0], self.dim_y)
        if torch.cuda.is_available():
            zero_state = zero_state.cuda()
        zero_state = F.sigmoid(self.enc_zero_dense(zero_state))

        peek_in = F.sigmoid(self.enc_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        _, final_state = self.enc_gru(torch.cat([x, peek], dim=2), zero_state)
        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)
        enc_res = self.enc_dense(cat_fw_bw)
        return enc_res


    def rnn_decoder(self, x, st):
        zero_state = torch.zeros(self.n_hidden, x.shape[0], self.dim_y)
        if torch.cuda.is_available():
            zero_state = zero_state.cuda()
        zero_state = F.sigmoid(self.dec_zero_dense(zero_state))

        peek_in = F.sigmoid(self.dec_peek_dense(st))
        peek = torch.reshape(peek_in.repeat([1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

        rnn_outputs, final_state = self.dec_gru(torch.cat([x, peek], dim=2), zero_state)
        dec_res = self.dec_dense(rnn_outputs)
        print("dec_res shape ", dec_res.shape)
        return dec_res
