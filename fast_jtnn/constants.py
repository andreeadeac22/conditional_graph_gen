import torch

##### conditional constants
num_prop = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
