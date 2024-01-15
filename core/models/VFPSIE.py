import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from core.models.utils import *
from core.models.submodel import *
from core.models.SCA import *

class ImageEncoder(nn.Module):
    def __init__(self, in_channel,init_chs=[24,36,54,72]):
        super(ImageEncoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            conv(in_channel, init_chs[0], 3, 2, 1),
            conv(init_chs[0], init_chs[0], 3, 1, 1)
        )

        self.pyramid2 = nn.Sequential(
            conv(init_chs[0],init_chs[1], 3, 2, 1),
            conv(init_chs[1], init_chs[1], 3, 1, 1)
        )

        self.pyramid3 = nn.Sequential(
            conv(init_chs[1], init_chs[2], 3, 2, 1),
            conv(init_chs[2], init_chs[2], 3, 1, 1)
        )

        self.pyramid4 = nn.Sequential(
            conv(init_chs[2], init_chs[3], 3, 2, 1),
            conv(init_chs[3], init_chs[3], 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4

    def get_fourth_feature(self, f3):
        f4 = self.pyramid4(f3)
        return f4


class EventEncoder(nn.Module):
    def __init__(self, in_channel,init_chs=[24,36,54,72]):
        super(EventEncoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            conv(in_channel, init_chs[0], 3, 2, 1),
            conv(init_chs[0], init_chs[0], 3, 1, 1)
        )

        self.pyramid2 = nn.Sequential(
            conv(init_chs[0], init_chs[1], 3, 2, 1),
            conv(init_chs[1], init_chs[1], 3, 1, 1)
        )

        self.pyramid3 = nn.Sequential(
            conv(init_chs[1], init_chs[2], 3, 2, 1),
            conv(init_chs[2], init_chs[2], 3, 1, 1)
        )

        self.pyramid4 = nn.Sequential(
            conv(init_chs[2], init_chs[3], 3, 2, 1),
            conv(init_chs[3], init_chs[3], 3, 1, 1)
        )

    def forward(self, event):
        f1 = self.pyramid1(event)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)

        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.attention = Symmetrical_Cross_Modal_Attention(dim=72, num_heads=4)
        self.convblock = nn.Sequential(
            conv(144, 144),
            ResBlock(144, 24),
            nn.ConvTranspose2d(144, 56, 4, 2, 1, bias=True)
        )

    def forward(self, I, E):
        AI, AE = self.attention(I, E)
        f_in = torch.cat([AI, AE], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            conv(164, 108),
            ResBlock(108, 24),
            nn.ConvTranspose2d(108, 38, 4, 2, 1, bias=True)
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        f_in = torch.cat([It_, I0_warp, E, up_flow0], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            conv(110, 72),
            ResBlock(72, 24),
            nn.ConvTranspose2d(72, 26, 4, 2, 1, bias=True)
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        f_in = torch.cat([It_, I0_warp, E, up_flow0], 1)

        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.attention =Symmetrical_Cross_Modal_Attention(dim=24, num_heads=1)
        self.convblock = nn.Sequential(
            conv(74, 48),
            ResBlock(48, 24),
            nn.ConvTranspose2d(48, 7, 4, 2, 1, bias=True)
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        AI, AE = self.attention(I0_warp, E)
        f_in = torch.cat([It_, AE + I0_warp,AI + E, up_flow0], 1)
        f_out = self.convblock(f_in)
        return f_out


class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.image_encoder = ImageEncoder(in_channel=3,init_chs=[24,36,54,72])
        self.event_encoder = EventEncoder(in_channel=10,init_chs=[24,36,54,72])
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()

    def forward(self, img0, evt0):

        intWidth = img0.shape[3] and evt0.shape[3]
        intHeight = img0.shape[2] and evt0.shape[2]

        intPadr = ((32 - intWidth%32)) % 32
        intPadb = ((32 - intHeight%32)) % 32

        img0_ = torch.nn.functional.pad(input=img0, pad=[0, intPadr, 0, intPadb], mode='replicate')
        evt0_ = torch.nn.functional.pad(input=evt0, pad=[0, intPadr, 0, intPadb], mode='replicate')

        I0_1, I0_2, I0_3, I0_4 = self.image_encoder(img0_)
        f0_1, f0_2, f0_3, f0_4 = self.event_encoder(evt0_)

        out4 = self.decoder4(I0_4, f0_4)
        up_flow0_4 = out4[:, 0:2]
        It_3_ = out4[:, 2:]

        out3 = self.decoder3(It_3_, I0_3, f0_3, up_flow0_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        It_2_ = out3[:, 2:]

        out2 = self.decoder2(It_2_, I0_2, f0_2, up_flow0_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        It_1_ = out2[:, 2:]

        out1 = self.decoder1(It_1_, I0_1, f0_1, up_flow0_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)

        img0_fusion = out1[:, 2:5]
        img0_fusion = torch.clamp(img0_fusion, 0, 1)

        weight = out1[:, 5:]
        weight = F.softmax(weight, dim=1)

        #frame inpainting
        img0_warp, mask = warp(img0_, up_flow0_1)
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        img0_warp[mask] = img0_fusion[mask]

        #weight fusion
        imgt_ = weight[:, 0:1, ...] * img0_fusion + weight[:, 1:2, ...] * img0_warp
        imgt_ = torch.clamp(imgt_, 0, 1)

        return imgt_[:,:,:intHeight,:intWidth]
