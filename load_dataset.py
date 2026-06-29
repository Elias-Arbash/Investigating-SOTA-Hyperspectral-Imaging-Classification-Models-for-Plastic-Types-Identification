"""
Investigating State of the Art Hyperspectral Imaging Classification Models for Plastic Types Identification" 
14th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing, WHISPERS 2024

EA - Helmholtz Institute Freiberg - HZDR

(Legacy loading system)
"""

import spectral as spi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spectral import *
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm

def read_data(datapath, paths, index1, index2):
    HSI = []
    GT = []
    
    for i, p in enumerate(paths):
        path_gt = str(datapath / str(p + '.png'))
        print(i)
        if i == 0:
            hsi, gt = read_HSI1(datapath,p,path_gt)
        if i == 1:
            hsi, gt = read_HSI2(datapath,p,path_gt)
        if i == 2:
            hsi, gt = read_HSI3(datapath,p,path_gt)
        if i == 3:
            hsi, gt = read_PE(datapath,p,path_gt)
        if i == 4:
            hsi, gt = read_PE(datapath,p,path_gt)
        if i == 5:
            hsi, gt = read_PE(datapath,p,path_gt)
        if i == 6:
            hsi, gt = read_PET(datapath,p,path_gt)
        if i == 7:
            hsi, gt = read_PP(datapath,p,path_gt)
        HSI.append(hsi[:,:,index1:index2])
        GT.append(gt)
    
    
    return HSI, GT

def read_HSI1(datapath,hsi_path,GT_path):
    #print(hsi_path,GT_path)
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)
    hsi = np.array(hsi, copy=True)
    hsi.setflags(write=1)
    hsi[np.isnan(hsi)] = 0
    
    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = cv2.imread(GT_path)
    GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    #print(np.unique(GT), GT.shape)
    hsi[GT == 7] = 0
    hsi[GT == 8] = 0
    
    GT[GT == 7] = 0
    GT[GT == 8] = 0
    
    return hsi ,GT

def read_HSI2(datapath,hsi_path,GT_path):
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr) / 65535.
    hsi = hsi[:,:,40:-50]
    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = cv2.imread(GT_path)
    GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    print(np.unique(GT), GT.shape)
    hsi = mask_hsi(hsi,GT)
    
    return hsi ,GT

def read_HSI3(datapath,hsi_path,GT_path):
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr) / 65535.
    hsi = hsi[:,:,40:-50]
    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = cv2.imread(GT_path)
    GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    print(np.unique(GT), GT.shape)
    hsi = mask_hsi(hsi,GT)
    
    return hsi ,GT

def read_PE(datapath,hsi_path,GT_path):
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr) / 65535.
    hsi = hsi[:,:,40:-50]

    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = np.zeros(hsi.shape[:2])
    # PE index is 6
    GT[GT>0] = 6
    GT[GT==0] = 6
    print(np.unique(GT), GT.shape)
    
    return hsi ,GT

def read_PET(datapath,hsi_path,GT_path):
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr) / 65535.
    hsi = hsi[:,:,40:-50]

    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = np.zeros(hsi.shape[:2])
    # PET index is 4
    GT[GT>0] = 4
    GT[GT==0] = 4
    print(np.unique(GT), GT.shape)
    
    return hsi ,GT

def read_PP(datapath,hsi_path,GT_path):
    header_file = str(datapath / str(hsi_path+ '.hdr')  )
    data_file = str(datapath /  str(hsi_path+ '.dat'))
    numpy_ndarr = envi.open(header_file, data_file)
    hsi = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr) / 65535.
    hsi = hsi[:,:,40:-50]

    print(np.min(hsi), np.max(hsi), hsi.shape)
    
    GT = np.zeros(hsi.shape[:2])
    # PP index is 1
    GT[GT>0] = 1
    GT[GT==0] = 1
    print(np.unique(GT), GT.shape)
    
    return hsi ,GT

def mask_hsi(hsi,gt):
    hsi[gt==0] = 0
    return hsi


def plot_images(images):
    """
    Plots 8 images in a 4 columns by 2 rows layout.

    Parameters:
    images (list of ndarray): List of 8 images to be plotted.

    """
    if len(images) != 8:
        raise ValueError("The input list must contain exactly 8 images.")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        if len(img.shape) > 2:
            ax.imshow(img[:,:,(-1,10,1)])
        else:
            
            # Define the class names
            class_names = ['Background', 'PP', 'Black Plastic', 'PVC', 'PET', 'ABS', 'PE']

            # Define the colors for each class
            colors = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # Example colors

            # Create a colormap
            cmap = ListedColormap(colors)

            # Create a normalization object
            norm = BoundaryNorm(np.arange(0, 8), cmap.N)

            # Plot the ground truth map
            ax.imshow(img, cmap=cmap, norm=norm, interpolation = 'None')
#            ax.imshow(img)
            ax.axis('on')  # Ensure axes are shown
            # Create the color bar


    plt.tight_layout()
    plt.show()
    

    
def normalize_hsi_bandwise_with_mask(hsi, mask):
    normalized_hsi = np.copy(hsi)
    for b in range(hsi.shape[2]):  # iterate over bands
        relevant_pixels = hsi[:, :, b][mask > 0]  # select only relevant pixels
        min_val = np.min(relevant_pixels)
        max_val = np.max(relevant_pixels)
        normalized_hsi[:, :, b] = (hsi[:, :, b] - min_val) / (max_val - min_val)
        # Ensure the background remains zero after normalization
        normalized_hsi[:, :, b][mask == 0] = 0
    return normalized_hsi

