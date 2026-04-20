"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@authors:
    Hyam Ali (hyam.omar-abbass-ali@univ-orleans.fr)
    Antoine Crosnier
    
"""

"""
 Description:
    This file contains the functions related to loading the data and extracting patches from the images.
"""




import scipy.io as sio
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def compute_global_metrics(lr_files, hr_files):
    all_intensity_values = []

    for lr_file in lr_files:
        corresponding_hr_file = lr_file.replace('_LR4', '')
        if corresponding_hr_file in hr_files:
            lr_image = sio.loadmat(lr_file)['radiance']
            hr_image = sio.loadmat(corresponding_hr_file)['radiance']
            #keep the channel separately
            all_intensity_values.append(lr_image.reshape(-1, lr_image.shape[2]))
            all_intensity_values.append(hr_image.reshape(-1, hr_image.shape[2]))
            #print("nb: channels =", lr_image.shape[2])

    all_intensity_values = np.concatenate(all_intensity_values)

    global_min = np.min(all_intensity_values, axis=0)
    global_max = np.max(all_intensity_values, axis=0)
    global_mean = np.mean(all_intensity_values, axis=0)
    global_median = np.median(all_intensity_values, axis=0)
    global_std = np.std(all_intensity_values, axis=0)
    #print("shape of all mean, std, min, max, median:", global_mean.shape, global_std.shape, global_min.shape, global_max.shape, global_median.shape)

    return all_intensity_values, global_min, global_max, global_mean, global_median, global_std

def convert_normalise_meanSTD(image, global_mean, global_std):
    min, max = image.min(), image.max()
    diff = max - min
    image = torch.tensor(image, dtype=torch.float32)
    image = (image - global_mean) / global_std
    norm_min, norm_max = image.min().item(), image.max().item()
    return image, min, max, diff, norm_min, norm_max



def extract_patches(image, patch_size, stride=16):
    img_h, img_w, bands = image.shape  
    patch_h, patch_w = patch_size

    patches = []
    #print(f'Extracting patches of size {patch_size} from image of size {(img_h, img_w)} with stride {stride}...')
    #print("img_h - patch_h + 1, img_w - patch_w + 1:", img_h - patch_h + 1, img_w - patch_w + 1)
    for i in range(0, img_h - patch_h + 1, stride):
        for j in range(0, img_w - patch_w + 1, stride):
            #print(i, j,img_h-patch_h + 1, img_w - patch_w + 1)
            patch = image[i:i + patch_h, j:j + patch_w, :]  
            patches.append(patch)
    print(f'Extracted {len(patches)} patches of size {patch_size} from image of size {(img_h, img_w)} with stride {stride}.')
    return np.array(patches)

def load_data_with_patches(args,data_path, global_mean=None, global_std=None,plot_images=False):
    patch_size = args.patch_size
    band= args.band_name
    mode= args.mode
    assert mode in ["lr-hr", "hr-sr"], "Mode must be either 'lr-hr' or 'hr-sr'"
    lr_data, hr_data, global_mean, global_std = load_normalise_data(data_path, band, global_mean, global_std)

    lr_patches = []
    hr_patches = []
    print(f'LR Data Shape: {lr_data.shape}' )
    print(f'HR Data Shape: {hr_data.shape}')
    if plot_images == True:
        to_plot = []
        sm = 0
    for lr_img, hr_img in zip(lr_data, hr_data):
        print(lr_img.shape, hr_img.shape)#H,W,C




        if plot_images:
            sm = sm + hr_img
            bands = [30, 100, 300]
            img1 = lr_img[:, :, bands]
            img1_vmin, img1_vmax = img1.min(), img1.max()
            img1 = (img1 - img1_vmin) / (img1_vmax - img1_vmin) if img1_vmax != img1_vmin else img1
            to_plot.append(img1)
            img2 = hr_img[:, :, bands]
            img2_vmin, img2_vmax = img2.min(), img2.max()
            img2 = (img2 - img2_vmin) / (img2_vmax - img2_vmin) if img2_vmax != img2_vmin else img2
            """fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img1)
            axes[0].set_title('LR Image')
            axes[0].axis('off')
            axes[1].imshow(img2)
            axes[1].set_title('HR Image')
            axes[1].axis('off')
            plt.show()"""
        sc_factor = args.sc_factor
        if mode == "lr-hr":
            if not(args.no_overlapp_patches):
                lr_img_patches = extract_patches(lr_img, patch_size)  # lr_img (spectral bands, H, W)
                hr_img_patches = extract_patches(hr_img, (patch_size[0] * sc_factor, patch_size[1] * sc_factor), stride=64)
            else:
                lr_img_patches = extract_patches(lr_img, patch_size, stride=patch_size[0])  # lr_img (spectral bands, H, W)
                hr_img_patches = extract_patches(hr_img, (patch_size[0] * sc_factor, patch_size[1] * sc_factor), stride=patch_size[0]*sc_factor)
        elif mode == "hr-sr":
            if not(args.no_overlapp_patches):
                hr_img_patches = extract_patches(hr_img, patch_size)  # hr_img (spectral bands, H, W)
                #cole batch size frome hr to create empty lr patches
                lr_img_patches = np.zeros((hr_img_patches.shape[0], patch_size[0] // sc_factor, patch_size[1] // sc_factor, hr_img_patches.shape[3]))
            else:
                hr_img_patches = extract_patches(hr_img, patch_size, stride=patch_size[0])  # hr_img (spectral bands, H, W)
                #cole batch size frome hr to create empty lr patches
                lr_img_patches = np.zeros((hr_img_patches.shape[0], patch_size[0] // sc_factor, patch_size[1] // sc_factor, hr_img_patches.shape[3]))
        lr_patches.extend(lr_img_patches)
        hr_patches.extend(hr_img_patches)
    if plot_images == True:
        nb = len(to_plot)
        print(f'Number of (non patched) images in the dataset: {nb}')
        H = int(torch.sqrt(torch.tensor(nb)).item())
        W = (nb)//H + (1 if (nb)%H !=0 else 0)
        if H !=1 and W !=1:
            fig, axes = plt.subplots(H, W, figsize=(15, 15))
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    if idx < nb:
                        axes[i, j].imshow(to_plot[idx])
                        axes[i, j].set_title(f'Image {idx+1}')
                        axes[i, j].axis('off')
                    else:
                        axes[i, j].axis('off')
            plt.show()
        else:
            fig, ax = plt.subplots(nb, 1, figsize=(5, 5 * nb))
            for i in range(nb):
                ax[i].imshow(to_plot[i])
                ax[i].set_title(f'Image {i+1}')
                ax[i].axis('off')
            plt.show()
    if plot_images:
        sm = sm / len(lr_data)
        #sm is of shape H,W,C
        #mean sm so that mean is of shape W,C
        mean_sm = sm.mean(axis=0)
        plt.figure(figsize=(10, 5))
        plt.imshow(mean_sm.T, cmap='gray')
        plt.title('Mean of HR Images across Height Dimension, depending of channel')
        #plt.axis('off')
        plt.xlabel('Width')
        plt.ylabel('Spectral Channels')
        plt.show()
        #compute the mean over the width dimension so mean_mean is of shape C
        mean_mean = mean_sm.mean(axis=0)
        #devide mean_sm by mean_mean to have the relative variation along the height
        mean_sm = mean_sm / mean_mean
        plt.figure(figsize=(10, 5))
        plt.imshow(mean_sm.T, cmap='gray')
        plt.title('Relative Variation of Mean HR Image across Height Dimension, depending of channel')
        #plt.axis('off')
        plt.xlabel('Width')
        plt.ylabel('Spectral Channels')
        plt.show()
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    print(f'LR Patch Shape: {lr_patches.shape}')
    print(f'HR Patch Shape: {hr_patches.shape}')
    
    return lr_patches, hr_patches, global_mean, global_std




def load_normalise_data(data_dir, BAND, global_mean=None, global_std=None):
    lr_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_newPipeline_clip_hyper_LR4.mat') and '_region_' in f and BAND in f ])
   
    hr_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_newPipeline_clip_hyper.mat') and '_region_' in f and BAND in f ])
    
    if global_mean is None or global_std is None:
        all_intensity_values, global_min, global_max, global_mean, global_median, global_std = compute_global_metrics(lr_files, hr_files)
        """
        output_csv = os.path.join(args.save_dir, args.save_prefix + '_global_metrics.csv')
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Global Min", global_min])
            writer.writerow(["Global Max", global_max])
            writer.writerow(["Global Mean", global_mean])
            writer.writerow(["Global Median", global_median])

        #plot_global_histogram(all_intensity_values, global_min, global_max, global_mean, global_median, global_std)
        """    
    #print(f"Using Global Mean: {global_mean}, Global Std: {global_std}")

    lr_data = []
    hr_data = []

    for lr_file in lr_files:
        corresponding_hr_file = lr_file.replace('_LR4', '')
        if corresponding_hr_file in hr_files:
            lr_image = sio.loadmat(lr_file)['radiance']
            hr_image = sio.loadmat(corresponding_hr_file)['radiance']
            lr_data.append(lr_image)
            hr_data.append(hr_image)

    """lr_data = np.array([img.numpy() for img in lr_data])
    hr_data = np.array([img.numpy() for img in hr_data])        """
    lr_data = np.array(lr_data)
    hr_data = np.array(hr_data)
    return lr_data, hr_data, global_mean, global_std
