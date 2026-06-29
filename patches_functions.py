"""
Investigating State of the Art Hyperspectral Imaging Classification Models for Plastic Types Identification" 
14th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing, WHISPERS 2024

EA - Helmholtz Institute Freiberg - HZDR
"""

import numpy as np
from tqdm import tqdm
from time import sleep
import cv2

def create_patches_with_labels2(hsi, mask, patch_size=7, fill_value_X=0, ignore_classes=[7, 8, 'cls1', 'cls2']):
    """
    Splits an HSI image into patches of the specified size and assigns labels based on the center pixel of the corresponding ground truth patch.
    
    Parameters:
        hsi (numpy.ndarray): The HSI image.
        mask (numpy.ndarray): The corresponding ground truth mask.
        patch_size (int, optional): Size of the patches to be extracted. Defaults to 7 (7x7 patches).
        fill_value_X (float, optional): Value to fill the remaining pixels in the last incomplete patches. Defaults to 0.
        ignore_classes (list, optional): List of classes to be ignored when creating patches. Defaults to [7, 8, 'cls1', 'cls2'].
    
    Returns:
        hsi_patches (numpy.ndarray): Array of extracted HSI patches.
        labels (numpy.ndarray): Array of labels corresponding to the center pixel of each HSI patch.
    """
    hsi_patches = []
    labels = []
    
    # Calculate the center offset
    center_offset = patch_size // 2

    # Extract image dimensions
    h, w, channels = hsi.shape
    
    # Pad HSI and mask to handle boundary conditions easily
    pad_hsi = np.pad(hsi, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), mode='constant', constant_values=fill_value_X)
    pad_mask = np.pad(mask, ((center_offset, center_offset), (center_offset, center_offset)), mode='constant', constant_values=fill_value_X)
    
    for i in range(h):
        for j in range(w):
            # Extract the patch
            patch_hsi = pad_hsi[i:i+patch_size, j:j+patch_size, :]
            
            # Get the label of the center pixel from the original mask (before padding)
            label = mask[i, j]
            
            # Check if the label is in the ignore_classes list
            if label not in ignore_classes:
                hsi_patches.append(patch_hsi)
                labels.append(label)
    
    # Convert lists to numpy arrays
    hsi_patches = np.array(hsi_patches)
    labels = np.array(labels)

    return hsi_patches, labels

def create_patches_with_labels(hsi_images, masks, patch_size=7, fill_value_X=0):
    """
    Splits HSI images into patches of the specified size and assigns labels based on the center pixel of the corresponding ground truth patch.

    Parameters:
        hsi_images (list): List of HSI images.
        masks (list): List of corresponding ground truth masks.
        patch_size (int, optional): Size of the patches to be extracted. Defaults to 7 (7x7 patches).
        fill_value_X (float, optional): Value to fill the remaining pixels in the last incomplete patches. Defaults to 0.

    Returns:
        hsi_patches (list): List of extracted HSI patches.
        labels (list): List of labels corresponding to the center pixel of each HSI patch.
    """
    hsi_patches = []
    labels = []
    
    # Calculate the center offset
    center_offset = patch_size // 2

    # Iterate through each HSI image and its corresponding mask
    for hsi, mask in zip(hsi_images, masks):
        # Extract image dimensions
        h, w, channels = hsi.shape
        
        # Pad HSI and mask to handle boundary conditions easily
        pad_hsi = np.pad(hsi, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), mode='constant', constant_values=fill_value_X)
        pad_mask = np.pad(mask, ((center_offset, center_offset), (center_offset, center_offset)), mode='constant', constant_values=fill_value_X)
        
        for i in range(h):
            for j in range(w):
                # Extract the patch
                patch_hsi = pad_hsi[i:i+patch_size, j:j+patch_size, :]
                
                # Get the label of the center pixel from the original mask (before padding)
                label = mask[i, j]
                
                hsi_patches.append(patch_hsi)
                labels.append(label)

    return hsi_patches, labels

def create_patches(hsi_images, masks, patch_size=7, fill_value_X=0):
    """
    Splits HSI images and corresponding masks into non-overlapping patches of the specified size.

    Parameters:
        hsi_images (list): List of HSI images.
        masks (list): List of corresponding masks.
        patch_size (int, optional): Size of the patches to be extracted. Defaults to 128 --> 128 x 128 x channels.
        fill_value_X (float, optional): Value to fill the remaining pixels in the last incomplete patches. Defaults to 0.

    Returns:
        hsi_patches (list): List of extracted HSI patches.
        mask_patches (list): List of extracted mask patches.
    """
    hsi_patches = []
    mask_patches = []
    
    # Iterate through each HSI image and its corresponding mask
    for hsi, mask in zip(hsi_images, masks):
        # Extract image dimensions
        h, w, channels = hsi.shape
        
        # Extract patches from the HSI image and mask
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch_hsi = hsi[i:i + patch_size, j:j + patch_size, :]
                patch_mask = mask[i:i + patch_size, j:j + patch_size]

                # Handle incomplete patches at the image boundaries
                if patch_hsi.shape[0] < patch_size or patch_hsi.shape[1] < patch_size:
                    pad_h = max(0, patch_size - patch_hsi.shape[0])
                    pad_w = max(0, patch_size - patch_hsi.shape[1])

                    # Pad the patch with the fill value
                    patch_hsi = np.pad(patch_hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=fill_value_X)
                    patch_mask = np.pad(patch_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=fill_value_X)

                hsi_patches.append(patch_hsi)
                mask_patches.append(patch_mask)

    return hsi_patches, mask_patches


def clipping_neg_pos(hscubes):
    """
    Clip all negative and positive values in the HSCubes array to 0 and 1, respectively.

    Parameters:
        hscubes (numpy.ndarray): PCB-Vision HS cubes array.

    Returns:
        clipped (numpy.ndarray): Clipped HSCubes array.
    """

    clipped = []

    # Iterate through each HSCube in the array
    for i in tqdm(range(len(hscubes))):
        sleep(0.001)  # Introduce a small delay to prevent excessive memory consumption

        # Copy the current HSCube
        img = hscubes[i].copy()

        # Replace negative values with 0
        img[np.where(img < 0.0)] = 0.0  

        # Replace positive values with 1
        img[np.where(img > 1.0)] = 1.0  

        # Reshape the clipped HSCube
        img = img[:, :, :] #?

        clipped.append(img)

        # Delete the copied HSCube to conserve memory
        del img

    print("Clipping data is complete. No more negative values.")
    
    return clipped

def slicing(hscubes, x):
    """
    Slices the given HSCubes array, removing the first `x` bands (channels) from each HSCube.

    Parameters:
        hscubes (numpy.ndarray): HSCubes array containing 224 bands (channels).
        x (int): Number of bands (channels) to remove from the beginning of each HSCube.

    Returns:
        sliced (numpy.ndarray): Sliced HSCubes array
    """

    sliced = []

    # Iterate through each HSCube in the array
    for i in tqdm(range(len(hscubes))):
        sleep(0.001)  # Introduce a small delay to prevent excessive memory consumption

        # Copy the current HSCube
        img = hscubes[i].copy()

        # Slice out the first `x` bands (channels)
        img = img[:, :, x:]

        sliced.append(img)

        # Delete the copied HSCube to conserve memory
        del img

    print("Skipping first " + str(x) + " bands")
    print("The HS cubes have " + str(224 - x) + " channels, the first " + str(x) + " are sliced out.")
    
    return sliced

