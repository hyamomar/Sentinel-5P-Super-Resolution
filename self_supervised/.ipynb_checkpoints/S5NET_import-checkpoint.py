"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains compatibility for importing the S5Net model to be able to evaluate it on the test set and compare it to the other models.
    The code below is adapted from https://github.com/alcarbone/S5P_SISR_Toolbox
"""
from torch import nn
import torch
import numpy as np

class S5Net(nn.Module):
    def __init__(self,n1,n2,n3,f1,f2,f3,c,dec_size,ratio,weights_deconv=None,weights_conv1=None,weights_conv2=None,weights_conv3=None,
                 biases_conv1=None,biases_conv2=None,biases_conv3=None):
        super(S5Net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(c, c, kernel_size=dec_size, stride=ratio, padding = dec_size-1-int(ratio/2)-int((dec_size-1)/2), bias=False)
        self.conv1 = nn.Conv2d(c, n1, kernel_size=f1, padding_mode='replicate', padding = 'same')
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding_mode='replicate', padding = 'same')
        self.conv3 = nn.Conv2d(n3, c, kernel_size=f3, padding_mode='replicate', padding = 'same')
        self.relu = nn.ReLU(inplace=True)
        
        if(weights_conv1!=None): 
            self.deconv1.weight.data = weights_deconv
            self.conv1.weight.data = weights_conv1
            self.conv2.weight.data = weights_conv2
            self.conv3.weight.data = weights_conv3
        
        if(biases_conv1!=None):
            self.conv1.bias.data = biases_conv1
            self.conv2.bias.data = biases_conv2
            self.conv3.bias.data = biases_conv3

    def forward(self, x, ratio, device):
        new_dim = [x.shape[2]*ratio,x.shape[3]*ratio]
        x = self.deconv1.forward(x)
        x = x[:,:,0:new_dim[0],0:new_dim[1]]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


import math
#model = S5Net(64,32,32,9,5,5,c,kernel_dec.shape[2],ratio)
def cubic(s, a=-0.5):
    if (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif ((abs(s) > 1) & (abs(s) <= 2)):
        return (a) * (abs(s) ** 3) - (5*a) * (abs(s) ** 2) + (8*a) * abs(s) - (4*a)
    return 0

def kernel(tc,ratio,interp_type):   
    if (interp_type == 'cubic'):
        q = 4
        M = q*(ratio-1)+2*math.floor((q+1)/2)-1
        k = np.zeros(M)
        x = -((M-1)/2)*tc/ratio
        for i in range(len(k)):
            k[i] = cubic(x,-0.5)
            x = x + tc/ratio
    else:
        print('Interpolation not available', file=sys.stderr)
        return None   
    return k

class BAND_S5Net(nn.Module):
    def __init__(self, band_name,ratio,train_data_min,train_data_max):
        super(BAND_S5Net, self).__init__()
        self.channel_start, self.channel_end = None,None
        self.train_data_min = train_data_min
        self.train_data_max = train_data_max
        if band_name == 'BAND2':
            self.channel_start, self.channel_end = 1,497
        elif band_name == 'BAND3':
            self.channel_start, self.channel_end = 498,994
        elif band_name == 'BAND4':
            self.channel_start, self.channel_end = 995,1491
        elif band_name == 'BAND5':
            self.channel_start, self.channel_end = 1492,1988
        elif band_name == 'BAND6':
            self.channel_start, self.channel_end = 1989,2485
        else:
            raise ValueError("Invalid band name, Band 7 and 8 not supported")
        self.S5Net_models = [None for _ in range(self.channel_start, self.channel_end+1)]
        self.ratio = ratio
        c = 1
        tc = 1
        ker = kernel(tc,ratio,'cubic')
        ker1 = np.zeros([ker.shape[0],1])
        ker1[:,0] = ker
        ker = ker1
        kernel_dec = np.matmul(ker,np.transpose(ker))    
        kernel_dec = np.transpose(kernel_dec)
        kernel_dec = torch.tensor(np.reshape(kernel_dec, (1,1,kernel_dec.shape[0],kernel_dec.shape[1])), dtype=torch.float64)

        for i in range(self.channel_start, self.channel_end+1):
            self.S5Net_models[i - self.channel_start] = S5Net(64,32,32,9,5,5,c,kernel_dec.shape[2],ratio)
    def load_internal_models(self, path_start):
        for i in range(self.channel_start, self.channel_end+1):
            #model are saved under band_{channel_number}/x{ratio}.pth
            path = path_start + f'band_{i}/x4.pth'
            self.S5Net_models[i - self.channel_start].load_state_dict(torch.load(path))
    def forward(self, x):
        B, C, H, W = x.shape
        #normalize input
        x = (x - self.train_data_min) / (self.train_data_max - self.train_data_min) *0.5

        #split according to the channels
        """channels = []
        for i in range(self.channel_start, self.channel_end+1):
            channels.append(x[:, i:i+1, :, :])
        #process each channel with its own S5Net model
        outputs = []
        for i, channel in enumerate(channels):
            model = self.S5Net_models[i]
            model.to(device)
            output = model.forward(channel, ratio, device)
            outputs.append(output)
        #concatenate the outputs along the channel dimension
        out = torch.cat(outputs, dim=1)
        return out"""
        device = x.device
        output = []
        for i in range(self.channel_start, self.channel_end+1):
            j = i - self.channel_start
            channel = x[:, j:j+1, :, :]
            #print(channel.shape,j,j+1,x.shape)
            model = self.S5Net_models[j]
            model.to(device)
            out_channel = model.forward(channel, self.ratio, device)
            output.append(out_channel)
        out = torch.cat(output, dim=1)
        out = out/0.5 * (self.train_data_max - self.train_data_min) + self.train_data_min
        #print input and output ranges
        x_min = torch.min(x).item()
        x_max = torch.max(x).item()
        out_min = torch.min(out).item()
        out_max = torch.max(out).item()
        x_avg = torch.mean(x).item()
        out_avg = torch.mean(out).item()
        #print(f"Input range: min={x_min}, max={x_max}, avg={x_avg}")
        #print(f"Output range: min={out_min}, max={out_max}, avg={out_avg}")
        return out