from torch import nn

from adaconv import *
from kernel_predictor import *
from ###

class AdaConvModel(nn.Module):
    def __init__(self, style_size, style_channels, kernel_size):
        super(AdaConvModel).__init__()
        self.encoder = ###

        style_in_shape = (self.encoder.out_channels, style_size // self.encoder.scale_factor, style_size 