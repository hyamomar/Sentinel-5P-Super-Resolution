"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the definitions of the different loss functions used for training and evaluation.
    Some of the code below is adapted from DeepInv library
"""


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
import matplotlib.pyplot as plt


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


#from deepinv.transform import ScalingTransform
from deepinv.loss import SupLoss, EILoss
from deepinv.loss.metric import mse





def sample_from(values, shape=(1,), dtype=torch.float32, device="cpu"):
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = torch.floor(N * torch.rand(shape, device=device, dtype=dtype)).to(
        torch.long
    )
    return values[indices]


def sample_downsampling_parameters(image_count, device, dtype, rates):
    downsampling_rate = sample_from(
        rates, shape=(image_count,), dtype=dtype, device=device
    )

    # The coordinates are in [-1, 1].
    center = torch.rand((image_count, 2), dtype=dtype, device=device)
    center = center.view(image_count, 1, 1, 2)
    center = 2 * center - 1

    return downsampling_rate, center


def get_downsampling_grid(shape, downsampling_rate, center, dtype, device):
    b, _, h, w = shape

    # Compute the sampling grid for the scale transformation
    u = torch.arange(w, dtype=dtype, device=device)
    v = torch.arange(h, dtype=dtype, device=device)
    u = 2 / w * u - 1
    v = 2 / h * v - 1
    U, V = torch.meshgrid(u, v, indexing="ij")
    grid = torch.stack([V, U], dim=-1)
    grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
    grid = (
        1 / downsampling_rate.view(b, 1, 1, 1).expand_as(grid) * (grid - center)
        + center
    )

    return grid


def alias_free_interpolate(x, downsampling_rate, interpolation_mode):
    xs = []
    for i in range(x.shape[0]):
        z = F.interpolate(
            x[i : i + 1, :, :, :],
            scale_factor=downsampling_rate[i].item(),
            mode=interpolation_mode,
            antialias=True,
        )
        z = z.squeeze(0)
        xs.append(z)
    return torch.stack(xs)


def padded_downsampling_transform(
    x, downsampling_rate, center, mode, padding_mode, antialiased
):
    shape = x.shape

    if antialiased:
        x = alias_free_interpolate(
            x, downsampling_rate=downsampling_rate, interpolation_mode=mode
        )

    grid = get_downsampling_grid(
        shape=shape,
        downsampling_rate=downsampling_rate,
        center=center,
        dtype=x.dtype,
        device=x.device,
    )
    return F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


class PaddedDownsamplingTransform(Module):
    def __init__(self, antialias, downsampling_rates, scaling_mc_factor=1):
        super().__init__()
        self.antialias = antialias
        self.downsampling_rates = downsampling_rates
        self.scaling_mc_factor = scaling_mc_factor
    def forward(self, x):
        downsampling_rate, center = sample_downsampling_parameters(image_count=x.shape[0]*self.scaling_mc_factor, device=x.device,dtype=x.dtype, rates=self.downsampling_rates)
        #print("scaling_mc_factor", self.scaling_mc_factor)
        #print("x shape before PaddedDownsamplingTransform:", x.shape)
        if self.scaling_mc_factor > 1:
            x = x.repeat(self.scaling_mc_factor,1,1,1)
        #print("x shape in PaddedDownsamplingTransform:", x.shape)
        x = padded_downsampling_transform(
            x,
            downsampling_rate=downsampling_rate,
            center=center,
            antialiased=self.antialias,
                mode="bicubic", padding_mode="reflection")
        #print("x shape after PaddedDownsamplingTransform:", x.shape)
        return x


def normal_downsampling_transform(x, downsampling_rate, mode, antialiased):
    xs = []
    for i in range(x.shape[0]):
        z = F.interpolate(
            x[i : i + 1, :, :, :],
            scale_factor=downsampling_rate,
            mode=mode,
            antialias=antialiased,
        )
        z = z.squeeze(0)
        xs.append(z)
    return torch.stack(xs)


class NormalDownsamplingTransform(Module):
    def __init__(self, antialias, downsampling_rates, scaling_mc_factor=1):
        super().__init__()
        self.antialias = antialias
        self.downsampling_rates = downsampling_rates
        self.scaling_mc_factor = scaling_mc_factor

    def forward(self, x):
        downsampling_rate = sample_from(self.downsampling_rates, shape=(self.scaling_mc_factor,), dtype=x.dtype, device=x.device )
        #downsampling_rate = downsampling_rate.item()
        if self.scaling_mc_factor == len(self.downsampling_rates):
            downsampling_rate = [torch.tensor(rate) for rate in self.downsampling_rates]
        if self.scaling_mc_factor == 1:
            x = normal_downsampling_transform(x,downsampling_rate=downsampling_rate.item(), mode="bicubic", antialiased=self.antialias)
            return x
        
        x = x.repeat(self.scaling_mc_factor,1,1,1)
        for i in range(self.scaling_mc_factor):
            #x = normal_downsampling_transform(x,downsampling_rate=downsampling_rate, mode="bicubic", antialiased=self.antialias)
            #x[i*x.shape[0]//self.scaling_mc_factor:(i+1)*x.shape[0]//self.scaling_mc_factor,:,:,:] = normal_downsampling_transform(x[i*x.shape[0]//self.scaling_mc_factor:(i+1)*x.shape[0]//self.scaling_mc_factor,:,:,:],downsampling_rate=downsampling_rate[i].item(), mode="bicubic", antialiased=self.antialias)
            o = normal_downsampling_transform(x[i*x.shape[0]//self.scaling_mc_factor:(i+1)*x.shape[0]//self.scaling_mc_factor,:,:,:],downsampling_rate=downsampling_rate[i].item(), mode="bicubic", antialiased=self.antialias)
            #pad o to have the same size as x
            pad_h = x.shape[2] - o.shape[2]
            pad_w = x.shape[3] - o.shape[3]
            #select random shift so the immage is not centered if it does a difference for the network
            pad_h1 = np.random.randint(0, pad_h + 1)
            pad_w1 = np.random.randint(0, pad_w + 1)
            #pad_h1 = pad_h // 2
            #pad_w1 = pad_w // 2
            #o = F.pad(o, (pad_h//2, pad_h - pad_h//2 , pad_w//2, pad_w - pad_w//2), mode='constant', value=0)
            o = F.pad(o, (pad_h1, pad_h - pad_h1 , pad_w1, pad_w - pad_w1), mode='constant', value=0)
            x[i*x.shape[0]//self.scaling_mc_factor:(i+1)*x.shape[0]//self.scaling_mc_factor,:,:,:] = o
        return x


class ScalingTransform(Module):
    def __init__(self, kind, antialias,scaling_mc_factor=1):
        super().__init__()
        #downsampling_rates = [0.75, 0.5]
        downsampling_rates = [0.75, 0.5, 0.25]
        if kind == "mirror":
            self.transform = PaddedDownsamplingTransform(antialias=antialias, downsampling_rates=downsampling_rates, scaling_mc_factor=scaling_mc_factor)
        elif kind == "no":
            self.transform = NormalDownsamplingTransform(antialias=antialias, downsampling_rates=downsampling_rates, scaling_mc_factor=scaling_mc_factor)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def forward(self, x):
        #print("x shape in ScalingTransform:", x.shape)
        #images are positive but not network output so we clip them
        x = nn.ReLU()(x)
        x = self.transform(x)
        """#PLOT THE IMAGES
        first_img = x[0,:,:,:].detach().cpu().numpy()
        #select chanel 30,50,100
        first_img = first_img[[30,50,100],:,:]
        #normalize the image for range 0-1
        min_val = first_img.min()
        max_val = first_img.max()
        first_img = (first_img - min_val) / (max_val - min_val)
        #print("first_img shape:", first_img.shape)
        plt.imshow(np.transpose(first_img, (1, 2, 0)))
        plt.title("First image after ScalingTransform")
        plt.axis('off')
        plt.show()"""
        


        #print("x shape after ScalingTransform:", x.shape)
        return x

class R2RLoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss(), eta=0.1, alpha=0.5):
        super(R2RLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.eta = eta
        self.alpha = alpha

    def forward(self, y, physics, model, **kwargs):
        pert = torch.randn_like(y) * self.eta

        y_plus = y + pert * self.alpha
        y_minus = y - pert / self.alpha

        output = model(y_plus, physics)

        return self.metric(physics.A(output), y_minus)


class R2REILoss(Module):
    def __init__(self, transform, sigma, no_grad=True, metric=None):
        super().__init__()
        self.T = transform
        self.sigma = sigma
        self.no_grad = no_grad
        if metric is None:
            metric = mse()
        self.metric = metric
        self.r2r_loss = R2RLoss(eta=self.sigma, alpha=0.5)

    def forward(self, *kargs, **kwargs):
        return self.r2r_loss(*kargs, **kwargs) + self.ei_loss(*kargs, **kwargs)

    # slightly modified for consistent input noise
    # base code available at https://github.com/deepinv/deepinv/blob/0b40ff5ac2f546987067465796ea55e5984d6967/deepinv/loss/ei.py
    def ei_loss(self, y, physics, model, **kwargs):
        epsilon1 = 0.5 * self.sigma * torch.randn_like(y)
        x1 = model(y + epsilon1, physics)

        if self.no_grad:
            with torch.no_grad():
                x2 = self.T(x1)
        else:
            x2 = self.T(x1)

        y2 = physics.A(x2)

        epsilon2 = 1.5 * self.sigma * torch.randn_like(y2)
        x3 = model(y2 + epsilon2, physics)

        return self.metric(x3, x2)














def mc_div(y1, y, model, physics, multiple_factor, tau, margin=0):
    y = y.view(1,y.shape[0],y.shape[1],y.shape[2],y.shape[3])
    y = y.repeat(multiple_factor,1,1,1,1)
    y1 = y1.view(1,y1.shape[0],y1.shape[1],y1.shape[2],y1.shape[3])
    y1 = y1.repeat(multiple_factor,1,1,1,1)
    assert margin is not None
    if margin == 0:
        b = torch.randn_like(y)
    else:
        ip_shape = (
            y.size(0),
            y.size(1),
            y.size(2),
            y.size(3) - 2 * margin,
            y.size(4) - 2 * margin,
        )
        b = torch.zeros_like(y)
        b[:, :, :, margin:-margin, margin:-margin] = torch.randn(
            *ip_shape, device=y.device, dtype=y.dtype) 
    y = y.view(-1,y.size(2),y.size(3),y.size(4))
    b = b.view(-1,b.size(2),b.size(3),b.size(4))
    y2 = physics.A(model(y + b * tau))
    y2 = y2.view(multiple_factor, -1, y2.size(1), y2.size(2), y2.size(3))
    b = b.view(multiple_factor, -1, b.size(1), b.size(2), b.size(3))
    out = b * (y2 - y1) / tau
    if margin != 0:
        out = out[:, :, :, margin:-margin, margin:-margin]
    out = out.mean(axis=(0,2,3,4))
    return out

def mc_div_correlated(y1, y, model, physics, multiple_factor, tau, margin=0):
    assert margin is not None
    y = y.view(1,y.shape[0],y.shape[1],y.shape[2],y.shape[3])
    y = y.repeat(multiple_factor,1,1,1,1)
    y1 = y1.view(1,y1.shape[0],y1.shape[1],y1.shape[2],y1.shape[3])
    y1 = y1.repeat(multiple_factor,1,1,1,1)
    if margin == 0:
        b = torch.randn_like(y)
    else:
        ip_shape = (
            y.size(0),
            y.size(1),
            y.size(2),
            y.size(3) - 2 * margin,
            y.size(4) - 2 * margin,
        )
        b = torch.zeros_like(y)
        b[:, :, :, margin:-margin, margin:-margin] = torch.randn(
            *ip_shape, device=y.device, dtype=y.dtype
        )
    y = y.view(-1,y.size(2),y.size(3),y.size(4))
    b = b.view(-1,b.size(2),b.size(3),b.size(4))
    # A A^T A
    y2 = physics.A(physics.A_adjoint(physics.A(model(y + b * tau))))
    #print("y",y.shape,"y1",y1.shape,"y2",y2.shape)
    y2 = y2.view(multiple_factor, -1, y2.size(1), y2.size(2), y2.size(3))
    b = b.view(multiple_factor, -1, b.size(1), b.size(2), b.size(3))
    out = b * (y2 - y1) / tau

    if margin != 0:
        out = out[:,:, :, margin:-margin, margin:-margin]

    out = out.mean(axis=(0,2,3,4))
    # mean to batch zise
    return out
def exact_divergence(y, model, physics, margin=0):
    y.requires_grad_(True)
    x_net = model(y)
    y2 = physics.A(x_net)
    div = torch.autograd.grad(
        outputs=y2, inputs=y,
        grad_outputs=torch.ones_like(y2),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    if margin != 0:
        div = div[:, :, margin:-margin, margin:-margin]
    div = div.mean(axis=(1,2,3))
    return div,x_net
def exact_divergence_correlated(y, model, physics, margin=0):
    y.requires_grad_(True)
    x_net = model(y)
    y2 = physics.A(physics.A_adjoint(physics.A(x_net)))
    div = torch.autograd.grad(
        outputs=y2, inputs=y,
        grad_outputs=torch.ones_like(y2),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    if margin != 0:
        div = div[:, :, margin:-margin, margin:-margin]
    div = div.mean(axis=(1,2,3))
    return div,x_net

class SureGaussianLoss(nn.Module):
    def __init__(self,args, tau=1e-2, margin=0, cropped_div=False, averaged_cst=False):

        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = args.noise_level ** 2
        self.tau = 1e-2 * 1e-7
        self.margin = margin
        self.cropped_div = cropped_div
        self.averaged_cst = averaged_cst
        self.mode = args.mode
        self.multiple_factor = args.mc_div_multiple_factor

    def forward(self, y, x_net, physics, model, **kwargs):
        #test if the model is in eval mode or training mode and olso if torch._no_grad is active
        if model.training or not(torch.is_grad_enabled()):
            if not(model.training):
                print("Warning: SURE loss in eval mode is being computed while gradients are disabled. Make sure this is intended.")
            y1 = physics.A(x_net)

            if self.mode == "lr-hr":
                div_fn = mc_div_correlated   
            else:  
                div_fn = mc_div             
            if self.sigma2 !=0:
                div = div_fn(y1, y, model, physics,self.multiple_factor, tau=self.tau, margin=self.margin)
                div = 2 * self.sigma2 * div

                mse = y1 - y
                if self.margin != 0:
                    mse = mse[:, :, self.margin:-self.margin, self.margin:-self.margin]
                mse = mse.pow(2).mean(axis=(1, 2, 3))

                loss_sure = mse + div
            else:
                mse = y1 - y
                if self.margin != 0:
                    mse = mse[:, :, self.margin:-self.margin, self.margin:-self.margin]
                mse = mse.pow(2).mean(axis=(1, 2, 3))
                loss_sure = mse
            return loss_sure
        else:
            #print("Computing SURE loss in eval mode with gradients enabled.")
            #compute exact divergence if in eval mode with gradients enabled
            if self.mode == "lr-hr":
                div_fn = exact_divergence_correlated
            else:
                div_fn = exact_divergence
            div, x_net = div_fn(y, model, physics, margin=self.margin)
            with torch.no_grad():
                y1 = physics.A(x_net)
                mse = y1 - y
                if self.margin != 0:
                    mse = mse[:, :, self.margin:-self.margin, self.margin:-self.margin]
                mse = mse.pow(2).mean(axis=(1, 2, 3))
                div = 2 * self.sigma2 * div
                loss_sure = mse + div
            #print("Sure loss in eval mode:", loss_sure, "x_net norm:", torch.norm(x_net))
            return loss_sure,x_net


            
class ClippedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        #self.mse = nn.MSELoss()

    def forward(self, obs, target):
        target_clipped = torch.clamp(target, min=0.0)
        #return self.mse(obs, target_clipped)
        a  =((obs-target_clipped)**2).mean(axis=(1,2,3))
        return a
    

class ProposedLoss(Module):
    def __init__(
        self,
        args,
        blueprint,
        sure_alternative,
        sure_cropped_div,
        sure_averaged_cst,
        sure_margin,
        physics,
    ):
        super().__init__()
        self.physics = physics
        self.mode = args.mode
        self.ssl_scalingTransform_mc_div_multiple_factor = args.ssl_scalingTransform_mc_div_multiple_factor
        if args.ssl_transform == "Scaling_Transforms":
            ei_transform = ScalingTransform(**blueprint[ScalingTransform.__name__])
        else:
            raise ValueError(f"Unknown transforms: {args.ssl_transform}")

        assert sure_alternative in [None, "r2r"]
        if sure_alternative == "r2r":
            loss_fns = [
                R2REILoss(
                    transform=ei_transform,
                    sigma=args.noise_level,
                    no_grad=args.ssl_stop_gradient,
                    metric=mse(),
                )
            ]
        else:
            sure_loss = SureGaussianLoss(
                args=args,
                cropped_div=sure_cropped_div,
                averaged_cst=sure_averaged_cst,
                margin=sure_margin
            )
            loss_fns = [sure_loss]

            equivariant_loss = EILoss(
                metric=ClippedMSE(),#metric=mse(),
                transform=ei_transform,
                no_grad=args.ssl_stop_gradient,
                weight=args.ssl_alpha_tradeoff,
                apply_noise=True,
            )
            

            loss_fns.append(equivariant_loss)
        self.loss_fns = loss_fns
        # NOTE: This could be done better.
        if sure_alternative == "r2r":
            self.compute_x_net = False
        else:
            self.compute_x_net = True

    def forward(self, x, y, model):
        ssl_scalingTransform_mc_div_multiple_factor = self.ssl_scalingTransform_mc_div_multiple_factor
        if model.training or not(torch.is_grad_enabled()):
            if self.compute_x_net:
                x_net = model(y)
            else:
                x_net = None

            total_loss = 0.0
            
            for loss_fn in self.loss_fns:
                loss_value = loss_fn(x=x, x_net=x_net, y=y, physics=self.physics, model=model)
                #print(f"{loss_fn.__class__.__name__}: {loss_value.shape}, requires_grad={loss_value.requires_grad}")                
                if isinstance(loss_value, torch.Tensor):
                    #print("loss ", loss_fn.__class__.__name__, loss_value.mean().item())
                    if ssl_scalingTransform_mc_div_multiple_factor == 1:
                        total_loss = total_loss + loss_value#.mean()
                    else:
                        if not(loss_fn.__class__.__name__ == "EILoss"):
                            #print("loss ", loss_fn.__class__.__name__, loss_value.mean().item())
                            #print("a",loss_value)
                            total_loss = total_loss + loss_value#.mean()
                        else:
                            #the ssl_scalingTransform_mc_div_multiple_factor multiplied the number of samples in the batch, so we need to reshape the loss
                            b = loss_value.shape[0]
                            f,b_ = ssl_scalingTransform_mc_div_multiple_factor, b//ssl_scalingTransform_mc_div_multiple_factor
                            loss_value = loss_value.view(f,b_).mean(axis=0)
                            #print("b",loss_value)
                            total_loss = total_loss + loss_value#.mean()
                else:

                    assert False
                    total_loss = total_loss + torch.tensor(loss_value, device=y.device, dtype=torch.float32)

            #print("total_loss:", total_loss, "requires_grad:", total_loss.requires_grad)
        else:
            x_net = None
            total_loss = 0.0
            #print(len(self.loss_fns))
            for loss_fn in self.loss_fns:
                try:
                    loss_value = None
                    r = loss_fn(x=x, x_net=x_net, y=y, physics=self.physics, model=model)
                    # r can be either a tensor (loss_value) or a tuple/list (loss_value, x_net)
                    if isinstance(r, (tuple)):
                        loss_value = r[0]
                        x_net = r[1]
                    else:
                        loss_value = r
                except:
                    if x_net is None:
                        #the first loss was not able to compute x_net, compute it now
                        with torch.no_grad():
                            x_net = model(y)
                    else:
                        #regenerate the error by calling again the loss function
                        r = loss_fn(x=x, x_net=x_net, y=y, physics=self.physics, model=model)
                        assert False, "This should not happen."
                    loss_value = loss_fn(x=x, x_net=x_net, y=y, physics=self.physics, model=model)
                #print(f"{loss_fn.__class__.__name__}: {loss_value.shape}, requires_grad={loss_value.requires_grad}")
                with torch.no_grad():
                    if isinstance(loss_value, torch.Tensor):
                        if ssl_scalingTransform_mc_div_multiple_factor == 1:
                            total_loss = total_loss + loss_value#.mean()
                        else:
                            if not(loss_fn.__class__.__name__ == "EILoss"):
                                #print("c",loss_value)
                                #print("loss ", loss_fn.__class__.__name__, loss_value)
                                total_loss = total_loss + loss_value#.mean()
                            else:
                                #the ssl_scalingTransform_mc_div_multiple_factor multiplied the number of samples in the batch, so we need to reshape the loss
                                b = loss_value.shape[0]
                                f,b_ = ssl_scalingTransform_mc_div_multiple_factor, b//ssl_scalingTransform_mc_div_multiple_factor
                                loss_value = loss_value.view(f,b_).mean(axis=0)
                                #print("d",loss_value)
                                #print("loss ", loss_fn.__class__.__name__, loss_value)
                                total_loss = total_loss + loss_value#.mean()
                    else:
                        assert False
                        total_loss = total_loss + torch.tensor(loss_value, device=y.device, dtype=torch.float32)

            #print("total_loss:", total_loss, "requires_grad:", total_loss.requires_grad)
        #print("final", total_loss)
        return total_loss





        


class Loss(Module):
    def __init__(
        self,
        args,
        physics,
        blueprint,
        sure_cropped_div,
        sure_averaged_cst,
        sure_margin,
    ):        
        super().__init__()
        if not args.ssl:
            if args.net_loss == 'MSE':
                def supervised_mse_loss(x, y, model):
                    assert x is not None
                    return torch.mean((model(y) - x) ** 2, dim=(1,2,3))
                self.loss = supervised_mse_loss
            else:
                assert False, f"Unknown net_loss: {args.net_loss}"
        else:
            if args.method == "proposed":
                self.loss = ProposedLoss(
                    args=args,
                    physics=physics,
                    blueprint=blueprint,
                    sure_cropped_div=sure_cropped_div,
                    sure_averaged_cst=sure_averaged_cst,
                    sure_margin=sure_margin,
                    **blueprint[ProposedLoss.__name__],
                )
            else:
                raise ValueError(f"Unknwon method: {args.method}")

    def forward(self, x, y, model):
        #if self.crop_fn is not None:
            #x, y = self.crop_fn(x, y, xy_size_ratio=self.xy_size_ratio)
        #print(f"Loss input shape: x {x.shape}, y {y.shape}")        
        loss = self.loss(x=x, y=y, model=model)
        #print("AFTER : inside loss function", loss.shape)
        #loss = loss.mean()
        return loss








def get_loss(args, physics):
    # NOTE: This is a bit of a mess.
    if args.partial_sure:
        if args.sure_margin is not None:
            sure_margin = args.sure_margin
        elif True:
            if args.partial_sure_sr:
                sure_margin = 2
            else:
                sure_margin = 0
    else:
        assert args.sure_margin is None
        sure_margin = 0

    blueprint = {}
    blueprint[Loss.__name__] = {
        #"crop_training_pairs": args.Loss__crop_training_pairs,
        #"crop_size": args.Loss__crop_size,
    }
    blueprint[ProposedLoss.__name__] = {
        "sure_alternative": args.ssl_sure_alternative,
    }

    blueprint[ScalingTransform.__name__] = {
        "kind": args.ssl_scalingTransform__kind,
        "antialias": args.ssl_scalingTransform__antialias,
        "scaling_mc_factor": args.ssl_scalingTransform_mc_div_multiple_factor,
    }

    sure_cropped_div = args.sure_cropped_div
    sure_averaged_cst = args.sure_averaged_cst

    loss = Loss(
        args=args,
        physics=physics,
        blueprint=blueprint,
        sure_cropped_div=sure_cropped_div,
        sure_averaged_cst=sure_averaged_cst,
        sure_margin=sure_margin,
        **blueprint[Loss.__name__],
    )

    return loss.to(args.device)

def get_debug_loss(args, physics):
    #invert ssl and sl for debugging purposes
    assert args.mode == "lr-hr"
    #copy args to new object
    new_args = copy.deepcopy(args)
    new_args.ssl = not args.ssl
    return get_loss(new_args, physics)
def get_debug_loss_similarity(args, physics):
    assert args.mode == "lr-hr"
    #copy args to new object
    new_args = copy.deepcopy(args)
    new_args.ssl = True
    l = get_loss(new_args, physics)
    p = l.loss.loss_fns.pop()  # remove equivariant loss
    print(f"Debug loss similarity: not using {p.__class__.__name__} loss for debugging.")
    for loss_fn in l.loss.loss_fns:
        print(f"Debug loss similarity: using {loss_fn.__class__.__name__} loss for debugging.")
    return l
