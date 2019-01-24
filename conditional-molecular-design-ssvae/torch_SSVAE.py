import torch
import torch.nn as nn

from torch_constants import *

class TorchSSVAE(nn.Module):
    def __init__(self, seqlen_x, dim_x, dim_y, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, beta=10000, char_set=[' ']):
        super(TorchSSVAE, self).__init__()
        self.seqlen_x, self.dim_x, self.dim_y, self.dim_z, self.dim_h, self.n_hidden, self.batch_size = seqlen_x, dim_x, dim_y, dim_z, dim_h, n_hidden, batch_size

        self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
        self.int_to_char = dict((i,c) for i,c in enumerate(char_set))

        self.predi_gru = torch.nn.GRU(dim_x, dim_h, num_layers=n_hidden, batch_first=True, bidirectional=True)
        self.predi_dense = torch.nn.Linear(dim_h * 2, dim_y)



    def forward(self, x_L, xs_L, y_L, x_U, xs_U):
        zero_state = torch.zeros(self.n_hidden *2, x_L.shape[0], self.dim_h)
        if torch.cuda.is_available():
            zero_state = zero_state.cuda()
        #print(x_L.shape)

        _, final_state = self.predi_gru(x_L, zero_state)

        #print("final_state", final_state.shape)

        cat_fw_bw = torch.cat([final_state[-1,:,:], final_state[-2,:,:]], dim=1)

        #print("cat_fw_bw.shape", cat_fw_bw.shape)

        predi_res = self.predi_dense(cat_fw_bw)

        return predi_res

    """
    def rnnpredictor(self, x, dim_h, dim_y, reuse=False):
        gru = torch.nn.GRU(x.shape[0], dim_h, num_layers=n_hidden, batch_first=True, bidirectional=True)
        zero_state = torch.zeros(x.shape[1], x.shape[2])
        _, final_state = gru(x, zero_state)
        print("final_state", final_state.shape)
        dense = torch.nn.Linear(final_state.shape[-1] * 2, dim_y)
        return dense(torch.concat(final_state[0], final_state[1]))
    """
