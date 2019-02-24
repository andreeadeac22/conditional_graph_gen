# pre-defined parameters
import torch

frac=0.5
beta=10000.
char_set=[' ','1','2','3','4','5','6','7','8','9','-','#','(',')','[',']','+','=','B','Br','c','C','Cl','F','H','I','N','n','O','o','P','p','S','s','Si','Sn']
data_uri='./data/ZINC_310k.csv'
save_uri='./zinc_model.ckpt'

ntrn=300000
frac_val=0.05
ntst=10000

dim_z = 100
dim_h = 250

n_hidden = 3
batch_size = 200

beta=10000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
