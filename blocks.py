import random
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
            self.upsample = partial(F.interpolate, scale_factor=2, mode='nearest')
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
            self.L2H = nn.Conv2d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
            self.H2L = nn.Conv2d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)

        def mask(self, hf, lf, alpha_in=0.25, alpha_out=0.25, order=True):
            mask_hf = torch.zeros_like(hf).cuda()
            mask_lf = torch.zeros_like(lf).cuda()
            c = hf.shape[1]
            hf_ch_out = int(c * (1 - alpha_out))
            lf_ch_out = c - hf_ch_out
            if order:
                index_hf = [i for i in range(hf_ch_out)]
            else:
                index_hf = random.sample(list(range(c)), hf_ch_out)
            index_lf = [i for i in range(c) if i not in index_hf]
            assert len(index_hf) == hf_ch_out
            assert len(index_lf) == lf_ch_out

            # [batch, channel, height, width]
            mask_hf[:,index_hf,:,:] = 1.
            mask_lf[:,index_lf,:,:] = 1.
            hf = hf * mask_hf
            lf = lf * mask_lf
            return hf, lf

        def forward(self, x, alpha_in, alpha_out):
            if self.type == 'first':
                hf = self.convh(x)
                lf = self.avg_pool(x)
                lf = self.convl(lf)
                hf, lf = self.mask(hf, lf, alpha_in=alpha_in, alpha_out=alpha_out)
                return hf, lf
            elif self.type == 'last':
                hf, lf = x
                return self.convh(hf) + self.convl(self.upsample(lf))
            else:
                hf, lf = x
                hf, lf = self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
                hf, lf = self.mask(hf, lf, alpha_in=alpha_in, alpha_out=alpha_out)
                return hf, lf
