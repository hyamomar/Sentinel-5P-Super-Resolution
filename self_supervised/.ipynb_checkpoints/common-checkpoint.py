"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains some utility definitions and functions, and also group all the imports of the different modules of the project.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#!pip install torchsummary
from torchsummary import summary
#!pip install tensorboard
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau



import os, glob

import csv
import matplotlib.pyplot as plt
import random
#!pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

#!pip install lpips
import lpips



from archithectures import *
from testing import *
from loading import *
from losses import *
from operator_ import *
from training import *




import subprocess



def select_least_used_gpu(coeff = [1,3]):
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")

        # Get memory usage
        mem_used = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        ).decode().strip().split('\n')
        mem_used = list(map(int, mem_used))

        # Get compute utilization
        compute_used = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader']
        ).decode().strip().split('\n')
        compute_used = list(map(int, compute_used))

        # Get total memory
        mem_total = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader']
        ).decode().strip().split('\n')
        mem_total = list(map(int, mem_total))

        # Compute relative usage
        mem_rel = [used / total for used, total in zip(mem_used, mem_total)]
        compute_rel = [used / 100 for used in compute_used]

        # Weighted score: 25% memory + 75% compute
        #scores = [(m + 3 * c) / 4 for m, c in zip(mem_rel, compute_rel)]
        #scores = [(c + 3 * m) / 4 for m, c in zip(mem_rel, compute_rel)]
        scores = [(coeff[0] * c + coeff[1] * m) / sum(coeff) for m, c in zip(mem_rel, compute_rel)]
        best_gpu = scores.index(min(scores))

        print(f"Selected GPU {best_gpu} with score {scores[best_gpu]:.2f}")
        torch.cuda.set_device(best_gpu)
        return torch.device(f"cuda:{best_gpu}")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Arguments:
    def __init__(self,
                band_name = "BAND6",
                ssl = False,
                ssl_sure = True, #False is unsure
                ssl_alpha_tradeoff = 1.0,
                ssl_transform =  "Scaling_Transforms",
                ssl_stop_gradient = True,
                ssl_sure_alternative = None,
                ssl_scalingTransform__kind = "no",#"mirror","no"
                ssl_scalingTransform__antialias = False,
                ssl_scalingTransform_mc_div_multiple_factor = 1,
                mode = "lr-hr",
                sc_factor = 4,
                noise_level = None,
                nepochs = 100,
                report_step = 5,
                batch_size = 1,
                mc_div_multiple_factor = 1,
                net_loss = 'MSE',
                net_opt = 'Adam',
                net_lr = 1e-2,
                net_name = '',
                save_dir = '',
                ftrain = '',
                ftest = '',
                fvalid = '',
                no_overlapp_patches = False,
                pretrain='',
                save_prefix='',
                device = "cpu",
                method = "proposed",
                patch_size = (64,64),
                partial_sure = True,
                sure_margin = 0,
                sure_cropped_div = False,
                sure_averaged_cst = True
                ):
        self.band_name = band_name
        self.ssl = ssl
        self.ssl_sure = ssl_sure
        self.ssl_alpha_tradeoff = ssl_alpha_tradeoff
        self.ssl_transform = ssl_transform
        self.ssl_stop_gradient = ssl_stop_gradient
        self.ssl_sure_alternative = ssl_sure_alternative
        self.ssl_scalingTransform__kind = ssl_scalingTransform__kind
        self.ssl_scalingTransform__antialias = ssl_scalingTransform__antialias
        self.ssl_scalingTransform_mc_div_multiple_factor = ssl_scalingTransform_mc_div_multiple_factor
        self.mode = mode
        self.sc_factor = sc_factor
        self.noise_level = noise_level
        self.nepochs = nepochs
        self.report_step = report_step
        self.batch_size = batch_size
        self.mc_div_multiple_factor = mc_div_multiple_factor
        self.net_loss = net_loss
        self.net_opt = net_opt
        self.net_lr = net_lr
        self.net_name = net_name
        self.save_dir = save_dir
        self.ftrain = ftrain
        self.ftest = ftest
        self.fvalid = fvalid
        self.no_overlapp_patches = no_overlapp_patches
        self.pretrain = pretrain
        self.save_prefix = save_prefix
        self.device = device
        self.method = method
        self.patch_size = patch_size
        self.partial_sure = partial_sure
        self.sure_margin = sure_margin
        self.sure_cropped_div = sure_cropped_div
        self.sure_averaged_cst = sure_averaged_cst
        if self.noise_level is None:
            self.noise_level = self.compute_noise_level(self.band_name)

    def compute_noise_level(self, band_name):
        band_data = {
            "BAND2": {"snr": 865.0440080503523, "global_mean_N": 1.0445112767449027e-07},
            "BAND3": {"snr": 1630.1901932130677, "global_mean_N": 2.2519189626330616e-07},
            "BAND4": {"snr": 3550.3207597241158, "global_mean_N": 5.022739016828739e-07},
            "BAND5": {"snr": 2277.1765307422875, "global_mean_N": 5.399677755428488e-07},
            "BAND6": {"snr": 2683.6019544451337, "global_mean_N": 5.475255866434212e-07},
            "BAND7": {"snr": 1093.9828705584187, "global_mean_N": 4.968771863421511e-08},
            "BAND8": {"snr": 875.2809156262713, "global_mean_N": 3.485563737221996e-08}
        }
    
        band_name = band_name.upper()
        if band_name not in band_data:
            raise ValueError(f"Unknown band name: {band_name}")
    
        snr = band_data[band_name]["snr"]
        global_mean_N = band_data[band_name]["global_mean_N"]
        sigma = global_mean_N / snr
        #sigma = 3.1026615109522616e-11 for band 6
        #sigma = 1.3772181712389384e-10
        return sigma
        

class CustomLoader:
    def __init__(self, data, mean,std, *args, **kwargs):
        print("len of data:", len(data))
        self.loader = DataLoader(data, *args, **kwargs)
        """self.mean = mean
        self.std = std"""
        """self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)"""
        self.mean = torch.tensor(mean,dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std,dtype=torch.float32).view(1, -1, 1, 1)

