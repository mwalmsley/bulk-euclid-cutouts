"""
Image processing used for the ELSE project
Courtesy Javier Acevedo Barroso

The script assumes that we have cutouts with the source centered, and that all the bands have the same zeropoint.
For the single band.
find reasonable values to clip the image, for that:

Calculate the background using the standard deviation at the corners of the image (but, it removes any corner with values much larger than the corner with the lower background) This is the low value.
Calculate the maximum value in a box (usually of 14 pixels) around the center. This helps against saturated stars. 
It should work as long as a our source is centered. This is the high value (times 1.3 because it looks a bit nicer imo)

Rescale the image by clipping and applying the scaling function (I prefer log for this):
For the composite band, we try to imitate DS9's approach.
Find reasonable values to clip the images, use the same value for all the bands. Here, we use percentils. I found reasonable defaults by trial-and-error, but there might be better ones depending on the size of the cutouts. Also, the high value is calculated using the same box as with single band, to avoid saturated stars.
Clip the image and renormalize it
Apply the scaling function (same log)

Apply bias-contrast scaling to the image
We first find reasonable values for the bias and contrast
The idea here is that the background should be visible, so that people can differentiate background from image easily, but the image must not be background dominated.
 color_bkg_level represents the final value of a background pixel. 
 This controls how bright is the background. We found the default by trial-and-error, but it should be around 0 anyway, since we still want to remove most of the noise.
Then, use the contrast-bias formula




Worked example


# I is VIS, from the ERO, resampled to 0.3 arcsec/pixel
    
fig, axes = plt.subplots(1,2,gridspec_kw={'wspace':0.02,
                                  'hspace':0.02}, squeeze=False)

                                #   This dict is the rescaled images, as a single dict
bands_dict = { #Rescales the zero points.
        'VIS':VIS,  # only used as single band so no need to rescale the zero point?
        'I':I * 10**(0.4 * (target_zeropoint - df_row.zero_point_VIS) ),  
        'H':H * 10**(0.4 * (target_zeropoint - df_row.zero_point_H) ),
        'Y':Y * 10**(0.4 * (target_zeropoint - df_row.zero_point_Y) )
        }

VIS_image, HYI_image = prepare_images(bands_dict,
                                f,
                                main_band = 'VIS',
                                composite_band = 'HYI'
                                )

axes[0,0].imshow(VIS_image, cmap = colormap)
axes[0,1].imshow(HYI_image, cmap = colormap)

"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:51:18 2024

@author: Javier Alejandro Acevedo Barroso
"""


def bands_to_array_via_zero_points(**bands):
    """
    Converts a dictionary of bands to an array using the zero points.
    
    Args:
        bands (dict): Dictionary of bands
        zero_points (dict): Dictionary of zero points
    
    Returns:
        array: Array of bands
    """

    DEFAULT_ZERO_POINTS = {
        'vis': 17.8,  # not the actual zero point due to e/s not ADU/s, manual hack
        'y': 24.3,
        'j': 24.5
    }

    return np.stack([bands[band] * 10**(0.4 * (24.0 - DEFAULT_ZERO_POINTS[band])) for band in bands],axis=-1)[:, :, ::-1]


# hmm - but the units are different for flux in VIS and YJH?



import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


# import os
# from os.path import join
# from astropy.io import fits
# from astropy.coordinates import SkyCoord

# pd.set_option('display.max_columns', None)
# plt.rcParams['image.origin'] = 'lower'

# def prepare_images(band_images, scale, main_band='VIS', composite_band='HYI'):
#     """
#     Prepares both main and composite bands images for visualization.
    
#     Args:
#         band_images (dict): Dictionary of band images
#         scale (function): Scaling function
#         main_band (str): Main band identifier (The band for the single band figure)
#         composite_band (str): Composite band identifier
    
#     Returns:
#         tuple: (main_image, composite_image) processed images
#     """
#     image = prepare_single_band(band_images[main_band],scale)
#     bands = list(composite_band)
#     composite_image = prepare_composite_band(np.stack([band_images[band] for band in bands],axis=-1), scale=scale)

#     return image, composite_image



def prepare_single_band(image, scale):
    """
    Prepares single band image for visualization.
    
    Args:
        image (array): Input single band image
        scale (function): Scaling function to apply
    
    Returns:
        array: Processed image ready for display
    """
    scale_min, scale_max = scale_val(image)
    # print(scale_min,scale_max)
    image = rescale_single_band_image(image, scale_min, scale_max, scale)
    image[np.isnan(image)] = np.nanmin(image)
    return image


def prepare_composite_band(images, scale, p_low=1, p_high=0.1, value_at_min=0, color_bkg_level=0.015):
    """
    Prepares multi-band composite image.
    
    
    Args:
        images (array): Input multi-band image array
        scale (function): Scaling function
        p_low (float): Lower percentile
        p_high (float): Higher percentile
        value_at_min (float): Minimum value threshold
        color_bkg_level (float): Background color level
    
    Returns:
        array: Processed composite image
    """
    composite_image = np.zeros_like(images,
                                dtype=float)
    
    scale_min, scale_max = get_value_range_asymmetric(images,p_low,p_high)
    assert scale_min < scale_max
    
    for i in range(images.shape[-1]):
        composite_image[:,:,i] = rescale_single_band_within_image(
                        images[:,:,i],
                        scale_min,
                        scale_max,
                        scale,
                        value_at_min,
                        color_bkg_level)
    return composite_image



def rescale_single_band_image(image, scale_min, scale_max, scale):
    """
    Rescales image values using specified bounds and scaling function.
    
    Args:
        image (array): Input image array
        scale_min (float): Minimum scaling value
        scale_max (float): Maximum scaling value
        scale (function): Scaling function to apply
    
    Returns:
        array: Rescaled image
    """
    factor = scale(scale_max - scale_min) # limits after applying scaling function
    image = image.clip(min=scale_min, max=scale_max)
    # print(factor)
    # factor can be negative: max - min is positive but small, and log10(smaller than 10) is negative
    factor = np.abs(factor)

    #J: I'm gonna go with this one since it solves the bright noise problem and seems to not hurt anything else.
    indices_below_min = np.where(image < scale_min)
    indices_between_vals = np.where((image >= scale_min) & (image < scale_max))
    indices_above_max = np.where(image >= scale_max)
    image[indices_below_min] = 0.0
    image[indices_above_max] = 1.0
    image[indices_between_vals] = scale(image[indices_between_vals]) / ((factor) * 1.0)
    return image


def rescale_single_band_within_image(image, scale_min, scale_max, scale, value_at_min=0, color_bkg_level=-0.05):
    """
    Rescales single band image with contrast and bias adjustment.
    
    Args:
        image (array): Input image array
        scale_min (float): Minimum scale value
        scale_max (float): Maximum scale value
        scale (function): Scaling function
        value_at_min (float): Minimum value threshold
        color_bkg_level (float): Background color level
    
    Returns:
        array: Rescaled and adjusted image
    """
    image = clip_normalize(image,scale_min,scale_max)
    image = scale(image)
    contrast, bias = get_contrast_bias_reasonable_assumptions(
                                                                max(value_at_min, scale_min),
                                                                color_bkg_level,
                                                                scale_min,
                                                                scale_max,
                                                                scale)
    return contrast_bias_scale(image, contrast, bias)




def clip_normalize(x, low=None, high=None):
    """
    Clips array values to specified range and normalizes to [0,1].
    
    Args:
        x (array): Input array to normalize
        low (float): Lower bound for clipping
        high (float): Upper bound for clipping
    
    Returns:
        array: Clipped and normalized array
    """
    x = np.clip(x, low, high)
    x = (x - low)/(high - low)
    return x 


def get_value_range_asymmetric(x, q_low=1, q_high=1):
    """
    Calculates asymmetric value range based on percentiles and central region.
    
    Args:
        x (array): Input image array
        q_low (float): Lower percentile value
        q_high (float): Higher percentile value
    
    Returns:
        tuple: (low, high) boundary values
    """
    low = np.nanpercentile(x, q_low)
    if x.shape[0] > 80:
        pixel_boxsize_low = np.round(np.sqrt(np.prod(x.shape) * 0.01)).astype(int)
    else:
        pixel_boxsize_low = 8
    xl, yl, _ = np.shape(x)
    xmin = int((xl) / 2. - (pixel_boxsize_low / 2.))
    xmax = int((xl) / 2. + (pixel_boxsize_low / 2.))
    ymin = int((yl) / 2. - (pixel_boxsize_low / 2.))
    ymax = int((yl) / 2. + (pixel_boxsize_low / 2.))
    # Also, the high value is calculated using the same box as with single band, to avoid saturated stars.
    high = np.nanpercentile(x[xmin:xmax,ymin:ymax], 100-q_high)
    return low, high


def background_rms_image(image, cb=10):
    """
    Calculates RMS of background using corner regions of the image.
    
    Args:
        image (array): Input image array
        cb (int): Size of corner boxes in pixels
    
    Returns:
        float: Standard deviation of background
    """
    xg, yg = np.shape(image)
    cut0 = image[0:cb, 0:cb]
    cut1 = image[xg - cb:xg, 0:cb]
    cut2 = image[0:cb, yg - cb:yg]
    cut3 = image[xg - cb:xg, yg - cb:yg]
    l = [cut0, cut1, cut2, cut3]
    while len(l) > 1:
        m = np.nanmean(np.nanmean(l, axis=1), axis=1)
        if max(m) > 5 * min(m):
            s = np.sort(l, axis=0)
            l = s[:-1]
        else:
            std = np.nanstd(l)
            return std
    std = np.nanstd(l)
    return std

def scale_val(image_array):
    """
    Determines scaling values for image visualization.

    find reasonable values to clip the image, for that:
    Calculate the background using the standard deviation at the corners of the image 
    (but, it removes any corner with values much larger than the corner with the lower background) This is the low value.
    Calculate the maximum value in a box (usually of 14 pixels) around the center. 
    This helps against saturated stars. It should work as long as a our source is centered. 
    This is the high value (times 1.3 because it looks a bit nicer imo)
    
    Args:
        image_array (array): Input image array
    
    Returns:
        tuple: (vmin, vmax) scaling values
    """
    #I'm assuming that the image is a galaxy cutout where the source is centered and not huge.
    if image_array.shape[0] > 173:
        box_size_vmin = np.round(np.sqrt(np.prod(image_array.shape) * 0.001)).astype(int)
        box_size_vmax = np.round(np.sqrt(np.prod(image_array.shape) * 0.01)).astype(int)
    else: 
        box_size_vmin = 5
        box_size_vmax = 14
        
    vmin = np.nanmin(background_rms_image(image_array,box_size_vmin))
    if vmin == 0:
        vmin += 1e-3              
    
    xl, yl = np.shape(image_array)
    xmin = int((xl) / 2. - (box_size_vmax / 2.))
    xmax = int((xl) / 2. + (box_size_vmax / 2.))
    ymin = int((yl) / 2. - (box_size_vmax / 2.))
    ymax = int((yl) / 2. + (box_size_vmax / 2.))
    vmax = np.nanmax(image_array[xmin:xmax, ymin:ymax])
    return vmin*1.0, vmax*1.3 #vmin is 1 sigma.  # use this values to clip the image (not simply percentiles)



def get_contrast_bias_reasonable_assumptions(value_at_min, bkg_color, scale_min, scale_max, scale):
    """
    Calculates contrast and bias parameters based on background assumptions.
    
    Args:
        value_at_min (float): Minimum value
        bkg_color (float): Background color level
        scale_min (float): Minimum scale value
        scale_max (float): Maximum scale value
        scale (function): Scaling function
    
    Returns:
        tuple: (contrast, bias) parameters
    """
    bkg_level = clip_normalize(value_at_min, scale_min, scale_max)
    bkg_level = scale(bkg_level)
    contrast = (bkg_color - 1) / (bkg_level - 1) # with bkg_level != 1 and bkg_color != 1
    bias = 1 - (bkg_level-1)/(2*(bkg_color-1))
    return contrast, bias


def contrast_bias_scale(x, contrast, bias):
    """
    Applies contrast and bias scaling to image.
    
    Args:
        x (array): Input array
        contrast (float): Contrast parameter
        bias (float): Bias parameter
    
    Returns:
        array: Contrast and bias adjusted array
    """
    x = ((x - bias) * contrast + 0.5 )
    x = np.clip(x, 0, 1)
    return x
