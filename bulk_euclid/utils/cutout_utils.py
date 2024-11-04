from bulk_euclid.utils import morphology_utils_ou_mer as m_utils  # try to keep this exactly like ou mer version

import numpy as np

# import Cutout2d
from astropy.nddata.utils import Cutout2D
# import WCS
from astropy.wcs import WCS

import cv2 ## new dependency
# pip install opencv-python


def make_composite_cutout_from_tiles(source, vis_im, nir_im, allow_radius_estimate=False):
    vis_cutout = m_utils.extract_cutout_from_array(vis_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    nisp_cutout = m_utils.extract_cutout_from_array(nir_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    
    return make_composite_cutout(vis_cutout, nisp_cutout)
    

def make_composite_cutout(vis_cutout, nisp_cutout, vis_q=100, vis_clip=99.85, nisp_q=1, nisp_clip=99.85):
    assert vis_cutout.shape == nisp_cutout.shape, f'vis shape {vis_cutout.shape}, nisp shape {nisp_cutout.shape}'
    assert vis_cutout.size < 19200**2, f'accidentally passed a whole tile, vis size {vis_cutout.size}'
    assert vis_cutout.shape != (19200, 19200), f'accidentally passed a whole tile, vis size {vis_cutout.size}'

    vis_flux_adjusted = m_utils.adjust_dynamic_range(vis_cutout, q=vis_q, clip=vis_clip)
    nisp_flux_adjusted = m_utils.adjust_dynamic_range(nisp_cutout, q=nisp_q, clip=nisp_clip)

    mean_flux = np.mean([vis_flux_adjusted, nisp_flux_adjusted], axis=0)
    
    vis_uint8 = m_utils.to_uint8(vis_flux_adjusted)
    nisp_uint8 = m_utils.to_uint8(nisp_flux_adjusted)
    mean_flux_uint8 = m_utils.to_uint8(mean_flux)

    im = np.stack([nisp_uint8, mean_flux_uint8, vis_uint8], axis=2)
    return im

def make_triple_cutout(short_wav_cutout, mid_wav_cutout, long_wav_cutout, short_q=100, mid_q=.2, long_q=.1, short_clip=99.85, mid_clip=99.85, long_clip=99.85):
    assert short_wav_cutout.shape == mid_wav_cutout.shape == long_wav_cutout.shape , f'short shape {short_wav_cutout.shape}, mid shape {mid_wav_cutout.shape}, long shape {long_wav_cutout.shape}'
    assert short_wav_cutout.size < 19200**2, f'accidentally passed a whole tile, short wavelength size {short_wav_cutout.size}'
    assert short_wav_cutout.shape != (19200, 19200), f'accidentally passed a whole tile, short wavelength size {short_wav_cutout.size}'

    short_flux_adjusted = m_utils.adjust_dynamic_range(short_wav_cutout, q=short_q, clip=short_clip)
    mid_flux_adjusted = m_utils.adjust_dynamic_range(mid_wav_cutout, q=mid_q, clip=mid_clip)
    long_flux_adjusted = m_utils.adjust_dynamic_range(long_wav_cutout, q=long_q, clip=long_clip)
    
    short_uint8 = m_utils.to_uint8(short_flux_adjusted)
    mid_uint8 = m_utils.to_uint8(mid_flux_adjusted)
    long_uint8 = m_utils.to_uint8(long_flux_adjusted)

    # RGB order is long, mid, short
    im = np.stack([long_uint8, mid_uint8, short_uint8], axis=2)
    return im


def make_lsb_cutout_from_tiles(source, vis_im, stretch, power, allow_radius_estimate=False):
    vis_cutout = m_utils.extract_cutout(vis_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    return make_lsb_cutout(vis_cutout, stretch, power)


def make_lsb_cutout(vis_cutout, stretch, power):
    vis_flux_lsb = gordon_scaling(vis_cutout, stretch, power)
    vis_uint8 = m_utils.to_uint8(vis_flux_lsb)
    return vis_uint8


def gordon_scaling(x, stretch, power):
    
    # original_min = x.min()
    original_min = 0
    original_max = x.max()
    
    # get alpha
    alpha_of_x = get_alpha(x, original_min, stretch, power)
    alpha_of_xmax = get_alpha(original_max, original_min, stretch, power)

    # apply arcinsh ratio
    x = np.arcsinh(alpha_of_x)/np.arcsinh(alpha_of_xmax)
    
    # clip
    # x = np.clip(x, original_min, original_max)
    
    # from astropy.convolution import convolve, Ring2DKernel
    # from photutils.segmentation import make_2dgaussian_kernel

    # kernel = Ring2DKernel(radius_in=0, width=2)
    # kernel = make_2dgaussian_kernel(20.0, size=3)  # FWHM = 3.0
    # x = convolve(x, kernel)
    
    x = np.clip(x, 0, np.percentile(x, 98))
    
    
    
    return x


"""Courtesy Tian Li"""

# First cover RGB image into LAB image
# Replace L channel in LAB space with the B channal (or VIS) in RGB spac
def replace_luminosity_channel(rgb_image: np.ndarray, rgb_channel_for_luminosity: int, desaturate_speckles: bool = False):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    # Replace L channel
    lab_image[:, :, 0] = rgb_image[:, :, rgb_channel_for_luminosity]
    # Convert back to RGB
    modified_rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    if desaturate_speckles:
        modified_rgb_image = desaturate_bright_pixels(modified_rgb_image)
    return modified_rgb_image


# this tends to give "blue speckles" from the blue channel/luminsity background
# desaturate individual pixels brighter than nearby pixels
# from astropy.stats import sigma_clipped_stats
def desaturate_bright_pixels(rgb_image: np.ndarray, threshold: int=150):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    # find highly saturated pixels
    saturation = hsv_image[:, :, 1]
    # print(np.isnan(saturation).mean())
    # saturation_nansafe = np.where
    # saturation_smoothed = cv2.medianBlur(saturation, 9)
    # saturation_diff = saturation - saturation_smoothed
    # desaturate highly saturated pixels
    # threshold = np.percentile(saturation, percentile)



    # mean, median, std = sigma_clipped_stats(hsv_image[:, :, 2], sigma=3.0)

    # threshold = threshold  # 0-255
    oversaturated = saturation > threshold
    hsv_image[:, :, 1] = np.clip(saturation, 0, threshold)
    lightness = hsv_image[:, :, 2]
    hsv_image[:, :, 2] = np.where(oversaturated, lightness * 0.5, lightness)

    # desaturation_factor = np.clip(saturation_diff / threshold, 0, 1)
    # hsv_image[:, :, 1] = hsv_image[:, :, 1] * (1 - desaturation_factor)
    # hsv_image[:, :, 2] = saturation_diff
    # Convert back to RGB
    modified_rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    # modified_rgb_image = saturation > threshold

    # bkg = hsv_image[:, :, 2] < (median - std)

    # print(threshold, (saturation.min(), saturation.max()), saturation.mean(), (saturation > threshold).mean())
    # return saturation > 170
    return modified_rgb_image





# this is basically a way to attempt to automatically set the "curves" tool from e.g. photoshop, where m sets the curve and MTF is the application of the curve. I will make a little comparison.
def MTF(x, desired_mean = 0.25):
    """Compute the Midtones Transfer Function (MTF) for given x and m."""


    x = np.clip(x, 0, np.percentile(x, 99))
    # x = np.arcsinh(x)
    # x = np.clip(x, 0, 1)  # image should already be normalized
    x = x / x.max()  # normalize to 0-1, maybe percentile clip

    m = find_m_for_mean(x, desired_mean)
    
    y = np.zeros_like(x)
    mask0 = (x == 0)
    maskm = (x == m)
    mask1 = (x == 1)
    mask_else = ~(mask0 | maskm | mask1)

    y[mask0] = 0  # 0 -> 0
    y[maskm] = 0.5 # 0.5 -> 0.5
    y[mask1] = 1  # 1 -> 1
    # ..and everything else gets curved
    x_else = x[mask_else]

    # shape of the curve
    numerator = (m - 1) * x_else
    denominator = (2 * m - 1) * x_else - m

    # apply the curve
    y[mask_else] = numerator / denominator

    return y  # curved values

def find_m_for_mean(normalized_image, desired_mean):
    x = np.mean(normalized_image)
    alpha = desired_mean
    return (x-alpha*x)/(x-2*alpha*x+alpha)
    
    
    
    
def get_alpha(x, original_min, stretch, power):

    
    # maximum not used
    return ( stretch * np.abs(x - original_min) / (1+stretch*np.abs(x-original_min)) ) ** power

# fits version of extract_cutout_from_array in moprhology_utils_ou_mer.py
def extract_cutout_from_fits(mosaic, mosaic_wcs, source, buff, allow_radius_estimate=False, enforce_shape=True):
    # pixel coordinates of cutout (r=x, c=y, not sure why this syntax)
    _, _, r1, r2, c1, c2 = m_utils.get_cutout_mosaic_coordinates(
        mosaic,
        source,
        buff,
        allow_radius_estimate
    )
    # cut mosaic (now with astropy)

    cutout = Cutout2D(
        mosaic, 
        position=(source['x_center'], source['y_center']), 
        size=(abs(r1-r2), abs(c1-c2)), 
        wcs=mosaic_wcs
    )
    data = cutout.data
    header = cutout.wcs.to_header()

    if enforce_shape:
        assert np.abs(data.shape[0] - data.shape[1]) <= 1, f'Shape mismatch: cutout of shape {data.shape} not square'
        # sometimes it is 1 pixel off, probably from rounding, and that's okay
    return data, header
