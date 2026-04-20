"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the definition of the operator that will be used in the training and testing of the model.
    Some of the code below is adapted from the DeepInv library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Import lr_scheduler module and ensure compatibility for LRScheduler attribute
import torch.optim.lr_scheduler as lr_scheduler
# expose ReduceLROnPlateau for backward compatibility
ReduceLROnPlateau = lr_scheduler.ReduceLROnPlateau

# Some versions of torch provide LRScheduler, older ones provide _LRScheduler.
# If LRScheduler is missing, alias it to an available base class or provide a minimal fallback.
if not hasattr(lr_scheduler, "LRScheduler"):
	if hasattr(lr_scheduler, "_LRScheduler"):
		lr_scheduler.LRScheduler = lr_scheduler._LRScheduler
	else:
		class LRScheduler:
			def __init__(self, optimizer, last_epoch=-1, verbose=False):
				self.optimizer = optimizer
				self.last_epoch = last_epoch
				self.verbose = verbose
			def state_dict(self):
				return {}
			def load_state_dict(self, state):
				pass
		lr_scheduler.LRScheduler = LRScheduler

from deepinv.physics import GaussianNoise,LinearPhysics


def get_gnyq(band_name):
    prefix = band_name[-1]
    if prefix == '2':
        return 0.36, 0.37
    elif prefix in ('3', '4', '5', '6'):
        return 0.74, 0.44
    elif prefix in ('7', '8'):
        return 0.20, 0.15
    else:
        return None, None
    

def generate_gaussian_kernel_torch(N, ratio, GNyq_x, GNyq_y, target_width_x=None):
    fcut = 1.0 / ratio
    std_x = torch.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * torch.log(torch.tensor(GNyq_x))))
    std_y = torch.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * torch.log(torch.tensor(GNyq_y))))
   
    if GNyq_x == 0.20 and GNyq_y == 0.15 and False: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        target_width_x = 8
        
    if target_width_x is not None:
        current_width_x = 2 * torch.sqrt(-2 * torch.log(torch.tensor(0.5)) * std_x**2)
        shrink_factor = current_width_x / target_width_x
        std_x /= shrink_factor

    ax = torch.arange(-(N - 1) / 2.0, (N + 1) / 2.0, dtype=torch.float64)
    gx = torch.exp(-0.5 * (ax / std_x)**2)
    gy = torch.exp(-0.5 * (ax / std_y)**2)
    Hdnew = torch.outer(gy, gx)

    return Hdnew


def apply_kaiser_window_torch(Hdnew, N, beta):
    Hdnew = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Hdnew))).real

    t_norm = torch.arange(-(N - 1) / 2.0, (N + 1) / 2.0, dtype=torch.float64) / (N - 1)
    t1n, t2n = torch.meshgrid(t_norm, t_norm, indexing='ij')
    t12 = torch.sqrt(t1n**2 + t2n**2)

    w1_np = np.kaiser(N, beta)
    t_np = t_norm.numpy()
    t12_np = t12.numpy()
    W_np = np.interp(t12_np.flatten(), t_np, w1_np).reshape(N, N)
    W_kaiser = torch.tensor(W_np, dtype=torch.float64)

    W_kaiser[t12 > t_norm[-1]] = 0
    W_kaiser[t12 < t_norm[0]] = 0

    psfK = Hdnew * W_kaiser
    psfK /= (psfK.sum() + 1e-12)

    return psfK


def convolution_spatial_zeropadding_torch(img, psf, padding_size):
    psf = psf / (psf.sum() + 1e-12)

    if img.ndim == 2:
        img_t = torch.tensor(img, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        psf_t = psf.unsqueeze(0).unsqueeze(0)
        out = F.conv2d(img_t, psf_t, padding=padding_size)
        return out.squeeze().numpy()

    elif img.ndim == 3:
        H, W, C = img.shape
        out = torch.zeros((H, W, C), dtype=torch.float64)
        for c in range(C):
            img_t = torch.tensor(img[:, :, c], dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            psf_t = psf.unsqueeze(0).unsqueeze(0)
            out[:, :, c] = F.conv2d(img_t, psf_t, padding=padding_size).squeeze()
        return out.numpy()

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def plot_hd_kaiser(Hdnew, psf):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Hdnew')
    plt.imshow(Hdnew.numpy(), cmap='jet')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('psf with Kaiser Window')
    plt.imshow(psf.numpy(), cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def resize_image_spatial_torch(img, ratio, GNyq_x, GNyq_y, N, beta, use_kaiser=True, target_width_x=None):
    
    Hdnew = generate_gaussian_kernel_torch(N, ratio, GNyq_x, GNyq_y, target_width_x=target_width_x)
    psf = apply_kaiser_window_torch(Hdnew, N, beta) if use_kaiser else Hdnew
    padding_size = N // 2
    blurred = convolution_spatial_zeropadding_torch(img, psf, padding_size)

    blurred = np.squeeze(blurred)

    lr = blurred[ratio // 2::ratio, ratio // 2::ratio, :]
    return lr

def resize_image_spatial_batch_torch(x, ratio, GNyq_x, GNyq_y, N=41, beta=0.5, use_kaiser=True, target_width_x=None):
    # x shape: (B, C, H, W)
    B, C, H, W = x.shape
    padding_size = N // 2

    psf = generate_gaussian_kernel_torch(N, ratio, GNyq_x, GNyq_y, target_width_x)
    #p = psf
    if use_kaiser:
        psf = apply_kaiser_window_torch(psf, N, beta)
    #plot_hd_kaiser(p, psf)
    #psf = psf.to(x.device)
    #psf = psf.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
    psf = psf.to(x.device).to(dtype=x.dtype)
    psf = psf.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
    psf = psf.expand(C,1,N,N)


    #x_reshaped = x.view(B * C, 1, H, W)
    #blurred = F.conv2d(x_reshaped, psf, padding=padding_size, groups=1)
    
    blurred = F.conv2d(x, psf, padding=padding_size, groups=C)

    #blurred = blurred.view(B, C, H, W)

    lr = blurred[:, :, ratio // 2::ratio, ratio // 2::ratio]
    return lr



class Downsampling(LinearPhysics):
    def __init__(self, rate, antialias, GNyq_x=None, GNyq_y=None):
        super().__init__()
        self.rate = rate
        self.antialias = antialias
        #print("GNyq_x:", GNyq_x, "GNyq_y:", GNyq_y)
        assert GNyq_x is not None and GNyq_y is not None, "GNyq_x and GNyq_y must be provided"
        self.GNyq_x = GNyq_x
        self.GNyq_y = GNyq_y

    
    def A(self, x):
        return resize_image_spatial_batch_torch(x, ratio=self.rate, GNyq_x=self.GNyq_x, GNyq_y=self.GNyq_y)


    def A_adjoint(self, y):    
        #print(" Using true adjoint (flipped PSF + upsampling)")
        B, C, H_lr, W_lr = y.shape
        H_hr, W_hr = H_lr * self.rate, W_lr * self.rate
        N = 41
        padding_size = N // 2
        beta = 0.5
    
        psf = generate_gaussian_kernel_torch(N, self.rate, self.GNyq_x, self.GNyq_y)
        psf = apply_kaiser_window_torch(psf, N, beta)
        psf = torch.flip(psf, dims=[0, 1])  
        psf = psf.to(y.device).to(dtype=y.dtype)
        psf = psf.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
        psf = psf.expand(C, 1, N, N)
    
        upsampled = torch.zeros((B, C, H_hr, W_hr), dtype=y.dtype, device=y.device)
        upsampled[:, :, self.rate // 2::self.rate, self.rate // 2::self.rate] = y
    
        x = F.conv2d(upsampled, psf, padding=padding_size, groups=C)
        return x


#custom noise model (to handle corelated or uncorelated noise if needed in future
class CustomNoiseModel():
    def __init__(self, sigma,mode,physics):
        self.noise_model = GaussianNoise(sigma=sigma)
        self.mode = mode
        self.physics = physics
        self.sigma = sigma
    def __call__(self, x):
        if self.sigma == 0.:
            return x
        if self.mode == "lr-hr":
            #corelated noise
            return x + self.physics.A(self.noise_model(torch.zeros_like(self.physics.A_adjoint(torch.zeros_like(x)))))
            #here A keeep the shape same as x
            #return x + self.physics.A(self.noise_model(torch.zeros_like(x)))
        else:
            #uncorelated noise
            return self.noise_model(x)
        

class PhysicsManager:
    def __init__(
        self,
        blueprint,
        task,
        mode,
        device,
        noise_level,
    ):
        if task == "sr":
            physics = Downsampling(antialias=True, **blueprint[Downsampling.__name__])
        else:
            raise ValueError(f"Unknown task: {task}")

        #physics.noise_model = GaussianNoise(sigma=noise_level)
        physics.noise_model = CustomNoiseModel(sigma=noise_level, mode=mode, physics=physics)
        # NOTE: These are meant to go.
        setattr(self, "task", task)
        setattr(physics, "task", task)
        setattr(physics, "__manager", self)

        self.physics = physics
        #print(f"Physics object created: {self.physics}")

    def get_physics(self):
        return self.physics

    def randomly_degrade(self, x, seed):
        # NOTE: Forking the RNG and setting the seed could be done all at once.
        preserve_rng_state = seed is not None
        with fork_rng(enabled=preserve_rng_state):
            if seed is not None:
                torch.manual_seed(seed)

            x = self.physics.A(x)
            x = self.physics.noise_model(x)
        return x

def get_physics(args):
    device = args.device
    blueprint = {}
    blueprint[PhysicsManager.__name__] = {
        "task": "sr",
        "mode": args.mode,
        "noise_level": args.noise_level,
    }


    blueprint[Downsampling.__name__] = {
        "rate": args.sc_factor,
        "GNyq_x": get_gnyq(args.band_name)[0],
        "GNyq_y": get_gnyq(args.band_name)[1],
    }

    physics_manager = PhysicsManager(
        blueprint=blueprint,
        device=device,
        **blueprint[PhysicsManager.__name__],
    )
    #print(f"Task: {args.task}")
    #print(f"Physics object created: {physics_manager.get_physics()}")
    #print("get_physics() was called!")
    return physics_manager.get_physics()