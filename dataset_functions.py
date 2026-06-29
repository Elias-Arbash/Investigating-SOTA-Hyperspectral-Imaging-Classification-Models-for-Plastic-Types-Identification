"""
Investigating State of the Art Hyperspectral Imaging Classification Models for Plastic Types Identification" 
14th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing, WHISPERS 2024

EA - Helmholtz Institute Freiberg - HZDR
"""

# Importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spectral as spi
from scipy import io
import cv2
import sys
import matplotlib as mpl
import seaborn as sns
import timeit
import logging
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from time import sleep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerBase
from spectral import *
from collections import Counter

logger = logging.getLogger(__name__)
logger.info('Elias Arbash: HSI/RGB PCBs Benchmark Dataset Functions')

def visualize(mask):
    """
    Show a PCB-vision mask in using specific colormap.
    
    Parameters:
        mask (numpy.ndarray): the 2D mask image.
    """
    
    colours = ['black','red', 'green', 'blue', 'Yellow']
    classes = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'Yellow'}
    cmap = []
    for i,x in enumerate(np.unique(mask)):
        cmap.append(classes[x])
    cmap = mpl.colors.ListedColormap(cmap)
    colormap = plt.imshow(mask, cmap=cmap, interpolation="none")
    plt.axis('off')

def read_hsi_mask(datapath, GTpath):
    """
    Reads an HSI and its corresponding mask from specified paths and returns the mask as a NumPy array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        GTpath (str): Path to the corresponding mask file.

    Returns:
        mask (numpy.ndarray): NumPy array containing the loaded mask.
    """
    # Convert the GTpath to POSIX format for compatibility
    GTpath2 = GTpath.as_posix()

    # Strip the extension
    spectral_file = str(GTpath2[:-4])  

    # Open the HSI data using ENVI and access the mask band using bipfile
    numpy_ndarr = envi.open(GTpath, spectral_file)
    y = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)

    mask = y[:, :, 0]

    return mask

def read_hsi_cube(datapath, Cubepath):
    """
    Reads an HSI cube from the specified path and returns it as a NumPy array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        Cubepath (str): Path to the HSI cube file.

    Returns:
        hsi_cube (numpy.ndarray): NumPy array containing the loaded HSI cube.
    """

    # Construct the paths to the header and spectral files
    header_file = str(datapath / Cubepath)
    # Remove extension
    spectral_file = str(datapath / Cubepath[:-4])  

    # Open the HSI data using ENVI and access the entire cube using bipfile
    numpy_ndarr = envi.open(header_file, spectral_file)
    hsi_cube = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)

    # Return the loaded HSI cube as a NumPy array
    return hsi_cube

    
def read_dataset(dataset_path):
    """This function reads the dataset.
    
    Parameters:
        - dataset_path (str): The path of the dataset folder.

    Returns:
        A tuple of 7 lists of 53 elements:
        
        - Hyperspectral Images (HSI) of the 53 PCB
        - HSI general segmentation masks
        - HSI mono segmentation masks
        - RGB images of the 53 PCB
        - RGB general segmentation masks
        - RGB mono segmentation masks
        - PCB Masks of the HSI
    """
    hsi_path = dataset_path + 'HSI/'
    rgb_path = dataset_path + 'RGB/'
    
    HSI = []
    HSI_seg_masks = []
    HSI_mono_masks = []
    RGB = []
    RGB_mono_masks = []
    RGB_general_masks = []
    PCB_Masks = []
    
    for i in tqdm(range(1,54)):
        sleep(0.001)
        datapath = Path( hsi_path + 'pcb' + str(i))
        Cubepath = "pcb" + str(i) + ".hdr"
        
        Maskpath = Path(hsi_path + 'General_masks/' + str(i) + ".HDR")
        HSI.append(read_hsi_cube(datapath, Cubepath))
        HSI_seg_masks.append(read_hsi_mask(datapath, Maskpath))
    
        Maskpath = Path(hsi_path + 'Monoseg_masks/mono' + str(i) + ".hdr")
        HSI_mono_masks.append(read_hsi_mask(datapath, Maskpath))
    
        RGBpath = rgb_path + str(i) + '.jpg'
        RGB.append(cv2.cvtColor(cv2.imread(RGBpath),cv2.COLOR_BGR2RGB))
    
        RGB_mono_masks_path = rgb_path + 'Monoseg/' + str(i) + '.png'
        RGB_mono_masks.append(np.array(Image.open(RGB_mono_masks_path)))
    
        RGB_general_masks_path = rgb_path + 'General/' + str(i) + '.png'
        RGB_general_masks.append(np.array(Image.open(RGB_general_masks_path)))
    
        maskspath = Path(hsi_path + 'PCB_Masks/'+str(i) + ".jpg")
        PCB_Masks.append(cv2.cvtColor(cv2.imread(str(maskspath) ),cv2.COLOR_BGR2GRAY))

    print("Dataset loading is complete.")
    return HSI, HSI_seg_masks, HSI_mono_masks, RGB, RGB_mono_masks, RGB_general_masks, PCB_Masks



# Function to train a model on a specific GPU
def set_gpu(gpu_id):
    """
    Function to set the current device to a specific GPU

    Parameters:
        gpu_id (int): The ID of the GPU to use

    Returns:
        torch.device: The selected GPU device
    """

    # Set the current device to the specified GPU ID
    torch.cuda.set_device(gpu_id)
    
    # Return the selected GPU device
    return torch.device("cuda")


def Generate_Training_data(training_list, HSI_cubes, seg_masks):
    """
    Generates augmented training data for the given HS cubes and their
    corresponding masks. The function for reading PCB-Vision HS
    training cubes and generating the training set.
    
    Parameters:
        training_list (list): A list of indices corresponding to PCB-Vision 
                              HS cubes and masks to be augmented.        
        HSI_cubes (list): A list of HS data cubes
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): A list of the augmented HS cubes 
        masks (list): A list of the augmented masks
        
    """
    cubes = []
    masks = []
    for i, ii in enumerate(training_list):
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        cube_aug, masks_aug = data_augmentation(HSI_cubes[ii-1], seg_masks[ii-1])
        for j in range(len(cube_aug)):
            cubes.append(cube_aug[j])
            masks.append(masks_aug[j])
        
        del cube_aug, masks_aug
        
    return cubes, masks

def Generate_data(data_list, HSI_cubes, seg_masks):
    """
    Reading PCB-Vision validation and testing HS cubes.
    It does not perform any augmentation
    
    Args:
        data_list (list): A list of indices corresponding to the HS cubes in 
                          PCB-Vision to be read.
        HSI_cubes (list): A list of the augmented HS cubes 
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): HS cubes
        masks (list): segmentation masks
    
    """
    cubes = []
    masks = []
    for i, ii in enumerate(data_list):
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        
    return cubes, masks

def evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes): # Yes Please
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    # Calculate pixel accuracy per class
    pixel_accuracy_per_class = true_positive_sum / (true_positive_sum + false_negative_sum)

    # Calculate pixel accuracy
    pixel_accuracy = np.sum(true_positive_sum) / np.sum(confusion_matrix_sum)

    # Calculate precision, recall, F1 score
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)
    recall = true_positive_sum / (true_positive_sum + false_negative_sum)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Calculate Intersection Over Union (IoU)
    iou = intersection_sum / union_sum

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_sum) / (np.sum(confusion_matrix_sum, axis=1) + np.sum(confusion_matrix_sum, axis=0))

    # Calculate Kappa coefficient
    total_pixels = np.sum(confusion_matrix_sum)
    observed_accuracy = np.sum(true_positive_sum) / total_pixels
    expected_accuracy = np.sum(true_positive_sum) * np.sum(confusion_matrix_sum, axis=1) / total_pixels**2
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    # Return the calculated evaluation metrics
    return confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class,   pixel_accuracy, iou, dice_coefficient, kappa

def reconstruct_prediction_image(test_image, test_GT, predictions, patch_size=11, fill_value_X=0, ignore_classes=[0]):
    """
    Reconstruct the prediction image from the predictions array.
    
    Parameters:
        test_image (numpy.ndarray): The test HSI image.
        test_GT (numpy.ndarray): The ground truth mask of the test image.
        predictions (numpy.ndarray): The predictions array.
        patch_size (int, optional): Size of the patches to be extracted. Defaults to 11 (11x11 patches).
        fill_value_X (float, optional): Value to fill the remaining pixels in the last incomplete patches. Defaults to 0.
        ignore_classes (list, optional): List of classes to be ignored when creating patches. Defaults to [0].

    Returns:
        prediction_map (numpy.ndarray): The reconstructed prediction image.
    """
    # Initialize the prediction map with zeros
    h, w = test_GT.shape
    prediction_map = np.zeros((h, w), dtype=np.int64)
    
    # Calculate the center offset
    center_offset = patch_size // 2

    # Pad HSI and mask to handle boundary conditions easily
    pad_hsi = np.pad(test_image, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), mode='constant', constant_values=fill_value_X)
    pad_mask = np.pad(test_GT, ((center_offset, center_offset), (center_offset, center_offset)), mode='constant', constant_values=fill_value_X)
    
    pred_idx = 0
    for i in range(h):
        for j in range(w):
            # Get the label of the center pixel from the original mask (before padding)
            label = test_GT[i, j]
            
            # Check if the label is in the ignore_classes list
            if label not in ignore_classes:
                # Place the prediction in the center pixel of the corresponding patch
                prediction_map[i, j] = predictions[pred_idx] + 1
                pred_idx += 1

    return prediction_map

# Define the gain_neighborhood_band function
def gain_neighborhood_band(x_train, band, band_patch, patch):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    
    # Center region
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    
    # Left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    
    # Right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    
    return x_train_band

# Function to compute class weights for cross entropy loss
def compute_class_weights(gt_list, num_classes):
    """
    Compute class weights based on the frequency of each class in the ground truth list.
    
    Args:
        gt_list (list of np.ndarray): List of ground truth masks.
        num_classes (int): Number of classes.
        
    Returns:
        np.ndarray: Array of class weights.
    """
    label_counts = Counter()
    
    for gt in gt_list:
        labels, counts = np.unique(gt, return_counts=True)
        for label, count in zip(labels, counts):
            label_counts[label] += count
    
    total_counts = sum(label_counts.values())
    class_weights = np.zeros(num_classes)
    
    for label in range(num_classes):
        if label in label_counts:
            class_weights[label] = total_counts / (num_classes * label_counts[label])
        else:
            class_weights[label] = 0  # Handle classes not present in the dataset
    
    return class_weights