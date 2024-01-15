import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# from basicsr.utils import get_root_logger
from einops import rearrange
import numbers
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

import sys
sys.path.append('.')
from .utils import *
from .Attention import *



"""
 Symmetrical_Cross_Modal_Attention
"""
class Symmetrical_Cross_Modal_Attention(nn.Module):
    def __init__(self, dim, num_heads , bias=False, LayerNorm_type='WithBias'):
        super(Symmetrical_Cross_Modal_Attention, self).__init__()
        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Mutual_Attention(dim, num_heads, bias)
        self.attn2 = Mutual_Attention(dim, num_heads, bias)

    def forward(self, image, event):

        assert image.shape == event.shape, 'the shape of image doesnt equal to event'

        b, c, h, w = event.shape
        norm_image = self.norm1_image(image)
        norm_event = self.norm1_event(event)
        
        enhanced_event = self.attn1(norm_image, norm_event)
        enhanced_image = self.attn2(norm_event,norm_image)

        return enhanced_image, enhanced_event





