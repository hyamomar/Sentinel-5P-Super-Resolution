"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the architecture elements for the proposed models.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn





def apply_mult(x,weight,bias):
    """if bias is None:
        return torch.matmul(x.permute(0,2,3,1), weight.squeeze()).permute(0,3,1,2)
    else:
        return torch.matmul(x.permute(0,2,3,1), weight.squeeze()).permute(0,3,1,2) + bias.view(1, -1, 1, 1)"""
    #avoid permute for speed
    # x: (batch, in_channels, H, W)
    # weight: (in_channels, out_channels)
    # compute pointwise (1x1) convolution via einsum: sum over input channels
    out = torch.einsum('bchw,co->bohw', x, weight)
    if bias is None:
        return out
    else:
        return out + bias.view(1, -1, 1, 1)
def apply_mult_SVD(x,U,S,bias):
    # x: (batch, in_channels, H, W)
    # U: (in_channels, rank)
    # S: (rank, out_channels)
    """x = x.permute(0,2,3,1)
    batch, H, W, in_channels = x.shape
    x = x.reshape(-1, x.size(-1))  # Shape (batch*H*W, in_channels)
    if bias is None:
        #return torch.chain_matmul(x.permute(0,2,3,1), U, S).permute(0,3,1,2)
        return torch.chain_matmul(x, U, S).view(batch, H, W, -1).permute(0,3,1,2)
    else:
        return torch.chain_matmul(x, U, S).view(batch, H, W, -1).permute(0,3,1,2) + bias.view(1, -1, 1, 1)"""
    #avoid permute for speed
    out = torch.einsum('bchw,cr,ro->bohw', x, U, S)
    if bias is None:
        return out
    else:
        return out + bias.view(1, -1, 1, 1)
class weight_no_compression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(weight_no_compression, self).__init__()
        # Custom weights for pointwise convolution
        #self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
        conv_1x1 = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             bias=False)
        conv_1x1.reset_parameters()
        self.weight = conv_1x1.weight
        #self.weight shape is (out_channels, in_channels, 1, 1)
        #reshape it to (in_channels, out_channels)
        self.weight = nn.Parameter(self.weight.view(out_channels, in_channels).t())
        #print("Weight matrix shape:", self.weight.shape)
        #self.plot_weight()
    def forward(self,x,bias):
        weight= self.weight
        x = apply_mult(x,weight,bias)
        return x
    def plot_weight(self):
        weight = self.weight.detach().cpu().numpy()
        plt.figure(figsize=(6,5))
        plt.imshow(weight, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Weight Matrix without Compression')
        plt.tight_layout()
        plt.show()
        #plot the svd decomposition
        U, S, Vt = np.linalg.svd(weight, full_matrices=False)
        #M = (U*S)@Vt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(U, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('U Matrix')
        plt.subplot(1,2,2)
        plt.imshow(Vt.T, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Vt.T Matrix')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15,5))
        plt.plot(S)
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()
        #plot cumulative explained variance
        explained_variance = np.cumsum(S**2) / np.sum(S**2)
        plt.figure(figsize=(15,5))
        plt.plot(explained_variance)
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.grid()
        plt.tight_layout()
        plt.show()


class weight_svd_compression(nn.Module):
    def __init__(self, in_channels, out_channels, rank =20):
        super(weight_svd_compression, self).__init__()
        # Custom weights for pointwise convolution
        #self.U = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
        #self.S = nn.Parameter(torch.randn(rank, out_channels) * 0.01)
        #for torchsummary name of parameters need to be "weight"
        conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                bias=False)
        conv_1x1.reset_parameters()
        #get svd of conv_1x1 weight
        weight_init = conv_1x1.weight
        #self.weight shape is (out_channels, in_channels, 1, 1)
        #reshape it to (in_channels, out_channels)
        weight_init = weight_init.view(out_channels, in_channels).t()
        weight_init = weight_init.detach().cpu().numpy()
        U_init, S_init, Vt_init = np.linalg.svd(weight_init, full_matrices=False)
        U_init = U_init[:, :rank]
        S_init = S_init[:rank]
        Vt_init = Vt_init[:rank, :]
        #renormalise U and Vt using sqrt(S)
        #shape of U is (in_channels, rank)
        #shape of S is (rank, rank)
        #shape of Vt is (rank, out_channels)
        #multiply columns of U by sqrt of singular values
        #print(U_init.shape, S_init.shape, Vt_init.shape)
        for i in range(rank):
            U_init[:, i] = U_init[:, i] * np.sqrt(S_init[i])
        #multiply rows of Vt by sqrt of singular values
        for i in range(rank):
            Vt_init[i, :] = Vt_init[i, :] * np.sqrt(S_init[i])
        #set the parameters
        with torch.no_grad():
            U = torch.tensor(U_init, dtype=torch.float32)
            S = torch.tensor(Vt_init, dtype=torch.float32)
            U = U.flatten()
            S = S.flatten()
        self.weight = nn.Parameter(torch.cat((U.detach(), S.detach()), dim=0))
        #self.weight = nn.Parameter(torch.randn(in_channels*rank+out_channels*rank) * 0.01)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.buffered_M = None
        #self.plot_weight()
    def get_U(self):
        return self.weight[:self.in_channels*self.rank].view(self.in_channels, self.rank)
    def get_S(self):
        return self.weight[self.in_channels*self.rank:].view(self.rank, self.out_channels)
    def forward(self,x,bias):
        #return(self.weight, self.bias)
        #reconstruct the weight
        if self.buffered_M is not None:
            """weight = self.buffered_M
            x = apply_mult(x,weight,bias)
            return x"""
            #using the buffered weight is not as computationally efficient as recomputing it with matrix multiplications in the right order
            x = apply_mult_SVD(x,self.get_U(),self.get_S(),bias)
            return x
        else:
            """U = self.get_U()
            S = self.get_S()
            weight = torch.matmul(U,S)
            x = apply_mult(x,weight,bias)"""
            x = apply_mult_SVD(x,self.get_U(),self.get_S(),bias)
            return x
    def plot_weight(self):
        U = self.get_U().detach().cpu().numpy()
        S = self.get_S().detach().cpu().numpy()
        M = np.matmul(U,S)
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(U, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('U Matrix')
        plt.subplot(1,3,2)
        plt.imshow(S, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('S Matrix')
        plt.subplot(1,3,3)
        plt.imshow(M, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Reconstructed Weight Matrix M=US')
        plt.tight_layout()
        plt.show()

        #compute the true svd of M to plot the singular values
        _, S_, _ = np.linalg.svd(M, full_matrices=False)
        plt.figure(figsize=(15,5))
        plt.plot(S_[:self.rank])
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()
    #when model is in eval mode, buffer the weight to avoid recomputing it each time
    def eval(self):
        super().eval()
        if not self.possible_buffer:
            U = self.get_U()
            S = self.get_S()
            self.buffered_M = torch.matmul(U,S)
    #when model is in train mode, unbuffer the weight to recompute it each time
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.buffered_M = None

class embed_to_values(nn.Module):
    def __init__(self, embd_size):
        super(embed_to_values, self).__init__()
        # Use individual layers instead of nn.Sequential for better torchsummary compatibility
        self.linear1 = nn.Linear(2 * embd_size, 50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50, 20)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(20, 1)
        
    def forward(self, embd):
        x = self.linear1(embd)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class weight_nn_compression(nn.Module):
    #predict matrix value with a small nn
    def __init__(self, in_channels, out_channels):
        super(weight_nn_compression, self).__init__()
        #get the index i,j as two inputs throught an embedding layer
        embd_size = 10
        #self.embedding_= custom_embedding(in_channels, out_channels, embd_size)
        self.weight = nn.Parameter(torch.randn(in_channels + out_channels, embd_size) * 0.01)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_to_values = embed_to_values(embd_size)
    def forward(self,x,bias):
        ins = self.weight[:self.in_channels, :]
        outs = self.weight[self.in_channels:, :]
        embd_i = ins  # Shape (in_channels, embd_size)
        embd_j = outs  # Shape (out_channels, embd_size)
        embd_i_exp = embd_i.unsqueeze(1).expand(-1, self.out_channels, -1)  # Shape (in_channels, out_channels, embd_size)
        embd_j_exp = embd_j.unsqueeze(0).expand(self.in_channels, -1, -1)  # Shape (in_channels, out_channels, embd_size)
        embd = torch.cat((embd_i_exp, embd_j_exp), dim=-1)  # Shape (in_channels, out_channels, 2*embd_size)
        embd = embd.view(-1, embd.size(-1))  # Shape (in_channels*out_channels, 2*embd_size)
        weight = self.emb_to_values(embd).squeeze(-1)  # Shape (in_channels* out_channels)
        weight = weight.view(self.in_channels, self.out_channels)  # Shape (in_channels, out_channels)
        return(apply_mult(x,weight,bias))
    def plot_weight(self):
        ins = self.weight[:self.in_channels, :]
        outs = self.weight[self.in_channels:, :]
        embd_i = ins  # Shape (in_channels, embd_size)
        embd_j = outs  # Shape (out_channels, embd_size)
        embd_i_exp = embd_i.unsqueeze(1).expand(-1, self.out_channels, -1)  # Shape (in_channels, out_channels, embd_size)
        embd_j_exp = embd_j.unsqueeze(0).expand(self.in_channels, -1, -1)  # Shape (in_channels, out_channels, embd_size)
        embd = torch.cat((embd_i_exp, embd_j_exp), dim=-1)  # Shape (in_channels, out_channels, 2*embd_size)
        weight = self.emb_to_values(embd).squeeze(-1).detach().cpu().numpy()  # Shape (in_channels, out_channels)
        plt.figure(figsize=(6,5))
        plt.imshow(weight, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Weight Matrix from NN Compression')
        plt.tight_layout()
        plt.show()
        #compute the svd of weight to plot the singular values
        _, S_, _ = np.linalg.svd(weight, full_matrices=False)
        plt.figure(figsize=(15,5))
        plt.plot(S_)
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()

class Custom_point_wise_conv(nn.Module):
    #redefined conv but with custom parameters
    def __init__(self, in_channels, out_channels, bias=False, compression="no"):
        super(Custom_point_wise_conv, self).__init__()
        # Custom weights for pointwise convolution
        #compression = "svd"#"no","svd","nn"
        self.compression_model = None
        if compression == "no":
            self.compression_model = weight_no_compression(in_channels, out_channels)
        elif len(compression)>=3 and compression[:3] == "svd":
            try:
                rank = int(compression[3:])
            except:
                rank = 20
                warn("No rank specified for svd compression, using default rank=20")
            if rank > min(in_channels, out_channels):
                self.compression_model = weight_no_compression(in_channels, out_channels)
                warn(f"Rank {rank} greater than min(in_channels, out_channels)={min(in_channels, out_channels)}. Using no compression instead.")
            else:
                self.compression_model = weight_svd_compression(in_channels, out_channels, rank=rank)
        elif compression == "nn":
            self.compression_model = weight_nn_compression(in_channels, out_channels)
            #raise a warning as summary doesn't work well with this model
            warn("Custom_point_wise_conv with nn compression not work well with torchsummary weight counting")
        else:
            raise ValueError("Unknown compression type")
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            #create an empty,trainable, null size weight to avoid errors in torchsummary (else the bias is not considered as trainable parameter)
            self.weight = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        #no need of conv because it's a 1x1 conv, use matrix multiplication instead
        return(self.compression_model(x,self.bias))


class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, upsample_scale=2, mode='bilinear',correct_relu = True,same_kernel = False,bias=False,compression="no"):
        super(DSC, self).__init__()
        self.same_kernel = same_kernel
        if self.same_kernel:
            self.depthwise_conv = nn.Conv2d(in_channels=1, 
                                            out_channels=depth_multiplier,  
                                            kernel_size=3,  
                                            stride=1,
                                            padding=1,
                                            groups=1,#num_spectral_bands,               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                            bias=bias)
        else:
            self.depthwise_conv = nn.Conv2d(in_channels=num_spectral_bands, 
                                            out_channels=num_spectral_bands * depth_multiplier,  
                                            kernel_size=3,  
                                            stride=1,
                                            padding=1,
                                            groups=num_spectral_bands,  
                                            bias=bias)
        
        """self.pointwise_conv = nn.Conv2d(in_channels=num_spectral_bands * depth_multiplier, 
                                        out_channels=out_channels,  
                                        kernel_size=1,  
                                        bias=bias)"""
        self.pointwise_conv = Custom_point_wise_conv(in_channels=num_spectral_bands * depth_multiplier,
                                                    out_channels=out_channels,
                                                    bias=bias or (correct_relu),
                                                    compression=compression)
        self.correct_relu = correct_relu
        if not(self.correct_relu):
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        
    def forward(self, x):
        #print("dfjnrgfjk")
        # shape (batch_size, num_spectral_bands, height, width)
        #!!!!!!!!!!!!!!!!!!!!!!!
        if self.same_kernel:
            shape_0,shape_1 = x.shape[0], x.shape[1]
            x = x.view(-1, 1, x.size(2), x.size(3))  # Ensure correct shape"""
        x = self.depthwise_conv(x)
        if self.same_kernel:
            x = x.view(shape_0, shape_1 * x.size(1), x.size(2), x.size(3))  # Reshape back
        x = self.pointwise_conv(x)
        if not(self.correct_relu):
            x = self.bn(x)
            x = self.relu(x)
        return x
    

class ReshapeLayer(nn.Module):
    def __init__(self, target):
        super(ReshapeLayer, self).__init__()
        self.target = target

    def forward(self, x):
        return x.view(-1,self.target, x.size(2), x.size(3))
class ImprovedDSC_2(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, num_layers=3, kernel_size=3, correct_relu = True, same_kernel = False,bias=False,compression="no"):
        super(ImprovedDSC_2, self).__init__()
        
        layers = []
        for ly in range(num_layers):
            if same_kernel:
                depthwise_conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=depth_multiplier,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2, 
                    groups=1,
                    bias=bias
                )
                layers.append(ReshapeLayer(1))  # Reshape before depthwise conv
                layers.append(depthwise_conv)
                layers.append(ReshapeLayer(num_spectral_bands * depth_multiplier))
            else:
                depthwise_conv = nn.Conv2d(
                    in_channels=num_spectral_bands,  
                    out_channels=num_spectral_bands * depth_multiplier,  
                    kernel_size=kernel_size,  
                    stride=1,
                    padding=kernel_size // 2, 
                    groups=num_spectral_bands,  
                    bias=bias
                )
                layers.append(depthwise_conv)
            """if correct_relu:
                layers.append(nn.ReLU())#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!test"""
            """pointwise_conv = nn.Conv2d(
                in_channels=num_spectral_bands * depth_multiplier,  
                out_channels=out_channels,  
                kernel_size=1,  
                bias=bias
            )"""
            pointwise_conv = Custom_point_wise_conv(
                in_channels=num_spectral_bands * depth_multiplier,
                out_channels=out_channels,
                bias=bias or (ly == num_layers - 1 and correct_relu),
                compression=compression
            )
            layers.append(pointwise_conv)
            if ly != num_layers - 1 or not(correct_relu):  # No ReLU after the last layer
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_layers(x)




class depth_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,compression="no"):
        super(depth_separable_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.internal_multiplier = ((out_channels // in_channels) + (out_channels % in_channels != 0))
        self.internal_channels = self.internal_multiplier * in_channels
        self.depthwise = nn.Conv2d(in_channels,
                                      self.internal_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=in_channels,
                                        bias= False)#only one biais
        self.pointwise = Custom_point_wise_conv(self.internal_channels, out_channels,bias=bias,compression=compression)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

