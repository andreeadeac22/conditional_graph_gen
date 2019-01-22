import torch

from constants import *

class TorchSSVAE(nn.Module):
    def __init__(self, seqlen_x, dim_x, dim_y, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, beta=10000, char_set=[' ']):
        super(TorchSSVAE, self).__init__()
        self.seqlen_x, self.dim_x, self.dim_y, self.dim_z, self.dim_h, self.n_hidden, self.batch_size = seqlen_x, dim_x, dim_y, dim_z, dim_h, n_hidden, batch_size
         self.beta = beta

         self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
         self.int_to_char = dict((i,c) for i,c in enumerate(char_set))

    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        
        return cost, objL, objU, objYpred_MSE

    def rnnpredictor(self, x, dim_h, dim_y, reuse=False):
        gru = torch.nn.GRU(dim_h, self.n_hidden, batch_first=True)
        rnn = torch.nn.RNN(self.n_hidden, 1)
