from math import ceil, floor

import torch
from torch import nn
from torch.nn import functional as F

from functools import partial
import pdb

class Attn_OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, alpha_in=0.5, alpha_out=0.5, type='normal'):
        super(Attn_OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type

        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        # hf_ch_in = in_channels
        # hf_ch_out = out_channels
        # lf_ch_in = in_channels
        # lf_ch_out = out_channels

        if type == 'first':
            self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias = False)
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.convl = nn.Conv2d(in_channels, lf_ch_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        elif type == 'last':
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.upsample = partial(F.interpolate, scale_factor, mode='nearest')
        else:
