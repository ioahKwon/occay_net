from math import ceil, floor

import torch
from torch import nn
from torch.nn import functional as F

class Attn_OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, alpha_in=0.5, alpha_out=0.5, type='normal'):
        super(Attn_OctConv, self).__init__()