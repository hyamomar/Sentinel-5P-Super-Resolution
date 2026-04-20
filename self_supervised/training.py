"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the training loop of the model
"""
from archithectures import S5_DSCR_S, S5_DSCR
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import csv
import matplotlib.pyplot as plt
from losses import get_loss, get_debug_loss, get_debug_loss_similarity
from operator_ import get_physics



def plot_hyperspectral_images_false_color_train(lr_img, hr_img, pred_img, idx, bands=[30, 50, 70], cmap='terrain'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    lr_color = lr_img[bands, :, :].transpose(1, 2, 0)
    hr_color = hr_img[bands, :, :].transpose(1, 2, 0)
    pred_color = pred_img[bands, :, :].transpose(1, 2, 0)


    lr_color = (lr_color - lr_color.min()) / (lr_color.max() - lr_color.min())
    hr_color = (hr_color - hr_color.min()) / (hr_color.max() - hr_color.min())
    pred_color = (pred_color - pred_color.min()) / (pred_color.max() - pred_color.min())

    lr_vmin, lr_vmax = lr_color.min(), lr_color.max()
    hr_vmin, hr_vmax = hr_color.min(), hr_color.max()
    pred_vmin, pred_vmax = pred_color.min(), pred_color.max()

    vmin = min(lr_vmin, hr_vmin, pred_vmin)
    vmax = min(lr_vmax, hr_vmax, pred_vmax)

    #axes[0].imshow(lr_color)
    axes[0].imshow(lr_color,  vmin=vmin, vmax=vmax)
    axes[0].set_title(f'LR Image (Bands {bands})')
    axes[0].axis('off')


    #axes[1].imshow(hr_color)
    axes[1].imshow(hr_color,  vmin=vmin, vmax=vmax)
    axes[1].set_title(f'HR Image (Bands {bands})')
    axes[1].axis('off')


    #axes[2].imshow(pred_color)
    axes[2].imshow(pred_color,  vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Predicted Image (Bands {bands})')
    axes[2].axis('off')

    fig.canvas.draw()
    return fig

def S5_DSCR_S_train(args,train_loader,valid_loader,num_bands,correct_relu = True,same_kernel = False, bias = False,compression="no",last_conv = False,min_val_stopping = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=num_bands, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias,
                            compression=compression,
                            last_conv=last_conv,
                            mean=mean,
                            std=std).to(args.device)
    model_name = 'DSC2'
    return generic_train(model,model_name,args,train_loader,valid_loader,num_bands,min_val_stopping=min_val_stopping)



def S5_DSCR_train(args,train_loader,valid_loader,num_bands,correct_relu = True, same_kernel = False, bias = False,compression="no",last_conv = False,min_val_stopping = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):

    model = S5_DSCR(
        in_channels=num_bands,
        out_channels=num_bands,
        num_spectral_bands=num_bands,
        depth_multiplier=3,
        num_layers=5,
        kernel_size=5,
        upsample_scale=4,
        correct_relu=correct_relu,
        same_kernel=same_kernel,
        bias=bias,
        compression=compression,
        last_conv=last_conv,
        mean=mean,
        std=std
    ).to(args.device)
    model_name = 'DSC_residual2'
    return generic_train(model,model_name,args,train_loader,valid_loader,num_bands,min_val_stopping=min_val_stopping)


def generic_train(model,model_name,args,train_loader,valid_loader,num_bands,min_val_stopping=False,debug_loss=True):
    device = args.device
    model = model.to(device)
    smother = False
    summary(model, input_size=(num_bands, 64, 64))

    log_dir = os.path.join(args.save_dir, args.save_prefix, model_name)
    writer_tensor = SummaryWriter(log_dir=log_dir)
    if args.net_opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.net_lr)
    elif args.net_opt == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.net_lr)
    elif args.net_opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.net_lr, momentum=0.90)
    else:
        raise ValueError(f"Unsupported optimizer: {args.net_opt}, use 'Adam', 'adadelta' or 'SGD'.")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1,eps = 0)

    train_losses, val_losses = [], []
    min_loss_val = float('inf')
    model_to_save = None
    stopped_epoch = -1
    mode = args.mode
    physics = get_physics(args)
    loss_fn = get_loss(args=args, physics=physics)
    debug_loss_fn = None
    if debug_loss:
        debug_loss_fn = get_debug_loss(args, physics)
    debug_loss_fn_sim = None
    if debug_loss:
        debug_loss_fn_sim = get_debug_loss_similarity(args, physics)
    for epoch in range(args.nepochs):
        #print the epoch number and the learning rate but not returning to the line so next prints will overwrite it
        print("epoch", epoch+1, "/", args.nepochs, " lr:", optimizer.param_groups[0]['lr'], "\r", end="")
        model.train()
        epoch_loss = 0
        mean, std = train_loader.mean, train_loader.std
        for batch_idx, (lr, hr) in enumerate(train_loader.loader):
            print("epoch", epoch+1, "/", args.nepochs, " batch", batch_idx+1, "/", len(train_loader.loader), "lr:", optimizer.param_groups[0]['lr'], "\r", end="")
            optimizer.zero_grad()

            if mode == "lr-hr":
                lr, hr = lr.to(device), hr.to(device)
                max_lr = lr.view(lr.size(0), -1).max(1)[0].view(-1, 1, 1, 1) 
                loss = loss_fn(x=hr, y=lr, model=model)
                loss = loss / (max_lr**2)
            else:
                hr = hr.to(device)
                max_hr = hr.view(hr.size(0), -1).max(1)[0].view(-1, 1, 1, 1) 
                loss = loss_fn(x=None, y=hr, model=model)
                loss = loss / (max_hr**2)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
                  
        val_loss = 0
        second_val_loss = False
        second_val = 0
        loss2 = None
        model.eval()
        #with torch.no_grad():
        debug_loss_val = 0
        debug_loss_val_sim = 0
        mean, std = valid_loader.mean, valid_loader.std
        for batch_idx,(lr, hr) in enumerate(valid_loader.loader):
            print("epoch", epoch+1, "/", args.nepochs, "Validation batch", batch_idx+1, "/", len(valid_loader.loader), "\r", end="")
            if mode == "lr-hr":
                lr, hr = lr.to(device), hr.to(device)
                max_lr = lr.view(lr.size(0), -1).max(1)[0].view(-1, 1, 1, 1) 
                loss = loss_fn(x=hr, y=lr, model=model)
                loss = loss / (max_lr**2)
                if second_val_loss:
                    loss2 = loss_fn(x=hr, y=lr, model=model)
                    loss2 = loss2 / (max_lr**2)
                if debug_loss:
                    debug_loss_batch = debug_loss_fn(x=hr, y=lr, model=model)
                    debug_loss_batch = debug_loss_batch / (max_lr**2)
                    debug_loss_val += debug_loss_batch.mean().item()
                    #print(max_lr**2)
                    debug_loss_batch_sim = debug_loss_fn_sim(x=hr, y=lr, model=model)
                    debug_loss_batch_sim = debug_loss_batch_sim / (max_lr**2)
                    debug_loss_val_sim += debug_loss_batch_sim.mean().item()
                    if smother:
                        debug_loss_batch = debug_loss_fn(x=hr, y=lr, model=model)
                        debug_loss_batch = debug_loss_batch / (max_lr**2)
                        debug_loss_val += debug_loss_batch.mean().item()
            else:  
                hr = hr.to(device)
                max_hr = hr.view(hr.size(0), -1).max(1)[0].view(-1, 1, 1, 1) 
                loss = loss_fn(x=None, y=hr, model=model)
                loss = loss / (max_hr**2)
                if second_val_loss:
                    loss2 = loss_fn(x=None, y=hr, model=model)
                    loss2 = loss2 / (max_hr**2)
                if debug_loss:
                    assert False, "Debug loss not existing for hr-sr mode"
            loss = loss.mean()
            val_loss += loss.item()
            if second_val_loss:
                loss2 = loss2.mean()
                second_val += loss2.item()
        if min_val_stopping:
            if val_loss/len(valid_loader.loader) < min_loss_val:
                min_loss_val = val_loss/len(valid_loader.loader)
                model_to_save = model.state_dict()
                stopped_epoch = epoch
        train_losses.append(epoch_loss / len(train_loader.loader))
        val_losses.append(val_loss / len(valid_loader.loader))
        if smother:
            debug_loss_val = debug_loss_val/2
        if not second_val_loss:
            if not debug_loss:
                print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")
            else:
                print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Debug Loss: {debug_loss_val/len(valid_loader.loader)},debug sim loss: {debug_loss_val_sim/len(valid_loader.loader)}")
        else:
            if not debug_loss:
                print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Second Validation Loss: {second_val/len(valid_loader.loader)}")
            else:
                print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Second Validation Loss: {second_val/len(valid_loader.loader)}, Debug Loss: {debug_loss_val/len(valid_loader.loader)},debug sim loss: {debug_loss_val_sim/len(valid_loader.loader)}")
        writer_tensor.add_scalar('Loss/Train', train_losses[-1], epoch)
        writer_tensor.add_scalar('Loss/Validation', val_losses[-1], epoch)
        scheduler.step(val_losses[-1])

    writer_tensor.close()
    if min_val_stopping:
        if stopped_epoch != args.nepochs - 1:
            print(f"Training stopped at epoch {stopped_epoch+1} with minimum validation loss: {min_loss_val}")
        else:
            print("Minimum of validation not reached during training.")

    try:
        if min_val_stopping and model_to_save is not None:
            torch.save(model_to_save, os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_hyperspectral_model.pth"))
            model.load_state_dict(model_to_save)
        else:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_hyperspectral_model.pth"))
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        with open(os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_losses.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
                writer.writerow([epoch, train_loss, val_loss])
        print('Loss file saved successfully.')
    except Exception as e:
        print(f"Error saving loss file: {e}")