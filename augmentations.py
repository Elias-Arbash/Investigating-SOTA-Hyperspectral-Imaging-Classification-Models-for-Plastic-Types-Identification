"""
Investigating State of the Art Hyperspectral Imaging Classification Models for Plastic Types Identification" 
14th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing, WHISPERS 2024

EA - Helmholtz Institute Freiberg - HZDR
"""

import numpy as np
import scipy.ndimage
import torch
from scipy.ndimage import gaussian_filter
import random

def spectral_shift(image, shift=1):
    """
    Description: Shift the spectral bands left or right by a few bands.
    Purpose: Mimics the effect of spectral variations and misalignments in the sensor.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return np.roll(image, shift, axis=2)
    return image
    


def spectral_smoothing(image, sigma=1):
    """
    Description: Apply a smoothing filter along the spectral dimension.
    Purpose: Reduces noise and simulates the effect of different sensor resolutions.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return gaussian_filter(image, sigma=[0, 0, sigma])
    return image


def spectral_noise(image, noise_level=0.05):
    """
    Description: Add Gaussian noise to the spectral bands.
    Purpose: Simulates sensor noise and improves the model's robustness to noisy data.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    return image + noise

def spectral_scaling(image, scale_factor=np.random.uniform(0.9, 1.1)):
    """
    Description: Scale the intensity of each spectral band by a random factor.
    Purpose: Simulates variations in illumination and sensor response.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    return image * scale_factor


def spectral_channel_dropping(image, drop_prob=0.99):
    """
    Description: Randomly drop some spectral channels.
    Purpose: Mimics sensor failure or missing data.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        drop_mask = np.random.rand(image.shape[2]) > drop_prob
        return image * drop_mask
    return image

def random_rotation(image, angle=np.random.uniform(-30, 30)):
    image = image.numpy() if torch.is_tensor(image) else image
    return scipy.ndimage.rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')


def random_translation(image, shift=np.random.uniform(-5, 5, size=2)):
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return scipy.ndimage.shift(image, shift=(shift[0], shift[1], 0), mode='reflect')
    elif image.ndim == 2:
        return scipy.ndimage.shift(image, shift=(shift[0], shift[1]), mode='reflect')
    return image

def random_flipping(image):
    image = image.numpy() if torch.is_tensor(image) else image
    if np.random.rand() > 0.05:
        image = np.flipud(image.copy())
    if np.random.rand() > 0.05:
        image = np.fliplr(image.copy())
    return image

def random_zoom(image, zoom_factor=None):
    if zoom_factor is None:
        zoom_factor = np.random.uniform(0.8, 1.2)
    
    image = image.numpy() if torch.is_tensor(image) else image
    
    if image.ndim == 3:
        h, w, c = image.shape
        zoomed_image = scipy.ndimage.zoom(image, (zoom_factor, zoom_factor, 1), order=1)
    elif image.ndim == 2:
        h, w = image.shape
        zoomed_image = scipy.ndimage.zoom(image, (zoom_factor, zoom_factor), order=1)
    else:
        return image

    # Handle zooming out (zoom_factor > 1.0)
    if zoom_factor > 1.0:
        # Crop the center of the zoomed image
        start_h = (zoomed_image.shape[0] - h) // 2
        start_w = (zoomed_image.shape[1] - w) // 2
        if image.ndim == 3:
            zoomed_image = zoomed_image[start_h:start_h + h, start_w:start_w + w, :]
        else:
            zoomed_image = zoomed_image[start_h:start_h + h, start_w:start_w + w]
    else:
        # Handle zooming in (zoom_factor < 1.0)
        pad_h = max((h - zoomed_image.shape[0]) // 2, 0)
        pad_w = max((w - zoomed_image.shape[1]) // 2, 0)
        
        if image.ndim == 3:
            # Add padding to the zoomed image
            zoomed_image = np.pad(zoomed_image,
                                  ((pad_h, h - zoomed_image.shape[0] - pad_h),
                                   (pad_w, w - zoomed_image.shape[1] - pad_w),
                                   (0, 0)),
                                  mode='reflect')
        else:
            zoomed_image = np.pad(zoomed_image,
                                  ((pad_h, h - zoomed_image.shape[0] - pad_h),
                                   (pad_w, w - zoomed_image.shape[1] - pad_w)),
                                  mode='reflect')

    return zoomed_image

def random_brightness(image, brightness_factor=np.random.uniform(0.8, 1.2)):
    image = image.numpy() if torch.is_tensor(image) else image
    return np.clip(image * brightness_factor, 0, 1)