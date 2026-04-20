"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the proposed architectures.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

from archi_com import *
import math




class S5_DSCR_S(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, upsample_scale=2, mode='bilinear',correct_relu = True,same_kernel = False,bias=False,compression="no",last_conv = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        super(S5_DSCR_S, self).__init__()
        self.interpolation = nn.Upsample(scale_factor=upsample_scale, mode='bicubic', align_corners=False)
        self.dsc_block = DSC(in_channels, out_channels, num_spectral_bands, depth_multiplier, correct_relu=correct_relu and not(last_conv), same_kernel=same_kernel, bias=bias, compression=compression)
        self.relu = nn.ReLU()
        self.correct_relu = correct_relu
        self.last_conv_b = last_conv
        self.last_conv = None
        if last_conv:
            self.last_conv = Custom_point_wise_conv(out_channels, out_channels,bias=True,compression=compression)
        self.mean = nn.Parameter(mean.view(1, -1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(1, -1, 1, 1), requires_grad=False)

    def forward(self, x, target_size=None, physics=None, **kwargs):
        #clip x to be compatible with equivariant loss
        x = nn.ReLU()(x)
        x = (x-self.mean)/self.std
        interpolated = self.interpolation(x)
        refined = self.dsc_block(interpolated)
        
        if target_size is not None:
            pass
            """interpolated = F.interpolate(interpolated, size=target_size, mode='bicubic', align_corners=False)
            refined = F.interpolate(refined, size=target_size, mode='bicubic', align_corners=False)"""
        else:
            refined = F.interpolate(refined, size=interpolated.shape[2:], mode='bicubic', align_corners=False)
        if self.last_conv_b:
            refined = self.last_conv(refined)
        output = refined + interpolated
        output = output * self.std + self.mean
        return output
    





class S5_DSCR(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, num_layers=3, kernel_size=3, upsample_scale=2, correct_relu = True, same_kernel = False,bias=False,compression="no",last_conv = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        super(S5_DSCR, self).__init__()
        self.interpolation = nn.Upsample(scale_factor=upsample_scale, mode='bicubic', align_corners=False)
        self.dsc_block = ImprovedDSC_2(in_channels, out_channels, num_spectral_bands, depth_multiplier, num_layers, kernel_size,correct_relu=correct_relu and not(last_conv), same_kernel=same_kernel, bias=bias, compression=compression)
        self.last_conv_b = last_conv
        self.last_conv = None
        if last_conv:
            self.last_conv = Custom_point_wise_conv(out_channels, out_channels,bias=True,compression=compression)
        self.mean = nn.Parameter(mean.view(1, -1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(1, -1, 1, 1), requires_grad=False)

    def forward(self, x, physics=None, **kwargs):
        #clip x to be compatible with equivariant loss
        x = nn.ReLU()(x)
        x = (x-self.mean)/self.std
        interpolated = self.interpolation(x)
        refined = self.dsc_block(interpolated)
        if self.last_conv_b:
            refined = self.last_conv(refined)
        output = refined + interpolated
        output = output * self.std + self.mean
        return output


class depth_separable_Unet(nn.Module):
    def __init__(self, num_spectral_bands, block_n_layers = 2,block_multiplier = 2, n_levels = 1, kernel_size=3,upsample_scale=2,bias=False,depth_multiplier=1,compression="no",bn_conv = False,bn_block = True,level_scale=2,last_biais= False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        super(depth_separable_Unet, self).__init__()
        layers_down = []
        l = num_spectral_bands
        channels_skip = []
        for i in range(n_levels):
            layers_down.append([])
            for j in range(block_n_layers):
                l_ = l * block_multiplier
                l_ = math.ceil(l_)
                print("layers_down : custom conv of size", l, "to", l_)
                b = bias or (not(bn_conv) and j!= block_n_layers -1) or (not(bn_block) and not(bn_conv) and j== block_n_layers -1)
                layers_down[-1].append(
                    depth_separable_conv(l, l_, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=b,compression=compression)
                )
                if bn_conv:
                    layers_down[-1].append(nn.BatchNorm2d(l_))
                elif bn_block and j == block_n_layers -1:
                    layers_down[-1].append(nn.BatchNorm2d(l_))
                layers_down[-1].append(nn.ReLU())
                if j == block_n_layers -1:
                    print("skip connection with", l_)
                    channels_skip.append(l_)
                l = l_
            layers_down[-1] = nn.Sequential(*layers_down[-1])
        layers_up = []
        for i in range(n_levels):
            layers_up.append([])
            for j in range(block_n_layers):
                if i!=0 and j ==0:
                    l_ = l + channels_skip[-i-1]
                    print("skip connection with", channels_skip[-i-1], " total ", l_)
                else:
                    l_ = l
                l__ = l / block_multiplier
                l__ = math.ceil(l__)
                print("layers_up : custom conv of size", l_, "to", l__)
                b = bias or (not(bn_conv) and j!= block_n_layers -1) or (not(bn_block) and not(bn_conv) and j== block_n_layers -1)
                layers_up[-1].append(
                    depth_separable_conv(l_, l__, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=b,compression=compression)
                )
                if bn_conv:
                    layers_up[-1].append(nn.BatchNorm2d(l__))
                elif bn_block and j == block_n_layers -1:
                    layers_up[-1].append(nn.BatchNorm2d(l__))
                layers_up[-1].append(nn.ReLU())
                l = l__
            layers_up[-1] = nn.Sequential(*layers_up[-1])
        self.layers_down = nn.ModuleList(layers_down)
        self.layers_up = nn.ModuleList(layers_up)
        self.pool = nn.MaxPool2d(level_scale)
        self.upsample = nn.Upsample(scale_factor=level_scale, mode='bilinear', align_corners=False)
        self.final_conv = Custom_point_wise_conv(l, num_spectral_bands,bias=last_biais,compression=compression)
        self.relu = nn.ReLU()
        self.num_spectral_bands = num_spectral_bands
        self.block_n_layers = block_n_layers
        self.block_multiplier = block_multiplier
        self.n_levels = n_levels
        self.kernel_size = kernel_size
        self.bias = bias
        self.depth_multiplier = depth_multiplier
        self.compression = compression
        self.interpolation = nn.Upsample(scale_factor=upsample_scale, mode='bicubic', align_corners=False)
        self.mean = nn.Parameter(mean.view(1, -1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(1, -1, 1, 1), requires_grad=False)
    def forward(self, x, physics=None, **kwargs):
        #clip x to be compatible with equivariant loss
        x = nn.ReLU()(x)
        x = (x-self.mean)/self.std
        interpolated = self.interpolation(x)
        x = interpolated
        skip_connections = []
        for i,block in enumerate(self.layers_down):
            x = block(x)
            if i != len(self.layers_down)-1:
                skip_connections.append(x.clone())
                x = self.pool(x)
        for i,block in enumerate(self.layers_up):
            #print("i=",i)
            if i !=0:
                x = self.upsample(x)
                skip = skip_connections.pop()
                #in case the size is odd, we need to pad the upsampled image
                if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                    x = F.pad(x, (0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2)))
                x = torch.cat((x, skip), dim=1)
            x = block(x)
        x = self.final_conv(x)
        output = x + interpolated
        output = output * self.std + self.mean
        return output

def unet_1M_x2(num_spectral_bands,upsample_scale=2, mean=torch.tensor(0.0), std=torch.tensor(1.0),full_bn=False):
    return depth_separable_Unet(num_spectral_bands,
        block_n_layers = 2,
        block_multiplier = 0.6,
        n_levels = 4,
        upsample_scale=upsample_scale,
        bias=True,
        depth_multiplier=1,
        compression="no",
        bn_conv = full_bn,
        bn_block = full_bn,
        level_scale=2,
        last_biais= True,
        mean=mean,
        std=std
    )
     
def unet_1M_x4(num_spectral_bands,upsample_scale=2, mean=torch.tensor(0.0), std=torch.tensor(1.0),full_bn=False):
    return depth_separable_Unet(num_spectral_bands,
        block_n_layers = 2,
        block_multiplier = 0.6,
        n_levels = 4,
        upsample_scale=upsample_scale,
        bias=True,
        depth_multiplier=1,
        compression="no",
        bn_conv = full_bn,
        bn_block = full_bn,
        level_scale=4,
        last_biais= True,
        mean=mean,
        std=std
    )

def unet_800k_x2(num_spectral_bands,upsample_scale=2, mean=torch.tensor(0.0), std=torch.tensor(1.0),full_bn=False):
    return depth_separable_Unet(num_spectral_bands,
        block_n_layers = 3,
        block_multiplier = 0.5,
        n_levels = 3,
        upsample_scale=upsample_scale,
        bias=True,
        depth_multiplier=1,
        compression="no",
        bn_conv = full_bn,
        bn_block = full_bn,
        level_scale=2,
        last_biais= True,
        mean=mean,
        std=std
    )
     
def unet_800k_x4(num_spectral_bands,upsample_scale=2, mean=torch.tensor(0.0), std=torch.tensor(1.0),full_bn=False):
    return depth_separable_Unet(num_spectral_bands,
        block_n_layers = 3,
        block_multiplier = 0.5,
        n_levels = 3,
        upsample_scale=upsample_scale,
        bias=True,
        depth_multiplier=1,
        compression="no",
        bn_conv = full_bn,
        bn_block = full_bn,
        level_scale=4,
        last_biais= True,
        mean=mean,
        std=std
    )

class BicubicUpsample(nn.Module):
    def __init__(self, scale_factor=4,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
        super(BicubicUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x, physics=None):
        #clip x to be compatible with equivariant loss
        x = nn.ReLU()(x)
        x = (x - self.mean) / self.std
        x = self.upsample(x)
        x = x * self.std + self.mean
        return x