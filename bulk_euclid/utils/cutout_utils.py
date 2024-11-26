from bulk_euclid.utils import morphology_utils_ou_mer as m_utils  # try to keep this exactly like ou mer version

import os
import numpy as np
from astropy.nddata.utils import Cutout2D
import cv2 ## new dependency
# pip install opencv-python
from PIL import Image
import logging

def save_jpg_cutouts(cfg, save_loc, vis_im: np.ndarray, y_im: np.ndarray=None, j_im: np.ndarray=None):
    # see lensing_cutout_colours.ipynb for a minimal example

    # flip to match fits if requested
    if cfg.use_fits_origin_for_jpg:
        vis_im = np.flipud(vis_im)
        if y_im is not None:
            y_im = np.flipud(y_im)
        if j_im is not None:
            j_im = np.flipud(j_im)

    if 'generic' not in save_loc:
        logging.debug(f'{save_loc} must include the string "generic" for renaming each cutout format e.g. foo_generic.jpg -> foo_sw_arcsinh_vis_y.jpg')
        raise AssertionError('save_loc must include the string "generic" for renaming each cutout format e.g. foo_generic.jpg -> foo_sw_arcsinh_vis_y.jpg')
    

    # data quality checks

    if np.nanmin(vis_im) >= np.nanmax(vis_im):
        logging.debug('vis band image is empty')
        raise AssertionError('vis band image is empty')

    if any(['vis_y' in x for x in cfg.jpg_outputs]):
        if y_im is None:
            logging.debug('Requested y colours but no y band image available')
            raise AssertionError('No y band image available')
        else:
            if np.nanmin(y_im) >= np.nanmax(y_im):
                logging.debug('y band image is empty')
                raise AssertionError('y band image is empty')
    if any(['vis_j' in x for x in cfg.jpg_outputs]) or any(['vis_y_j' in x for x in cfg.jpg_outputs]):
        if j_im is None:
            logging.debug('Requested j colours but no j band image available')
            raise AssertionError('No j band image available')
        else:
            if np.nanmin(j_im) >= np.nanmax(j_im):
                logging.debug('j band image is empty')
                raise AssertionError('j band image is empty')

    ### GZ Euclid arcsinh processing ###

    if 'gz_arcsinh_vis_y' in cfg.jpg_outputs:
        cutout = make_composite_cutout(vis_im, y_im, vis_q=100, nisp_q=0.2)
        save_image_wrapper(cutout, save_loc.replace('generic', 'gz_arcsinh_vis_y'), quality=cfg.jpg_quality)
                    
    if 'gz_arcsinh_vis_only' in cfg.jpg_outputs:
        cutout = m_utils.make_vis_only_cutout(vis_im, q=100)
        save_image_wrapper(cutout, save_loc.replace('generic', 'gz_arcsinh_vis_only'), quality=cfg.jpg_quality)

    if 'gz_arcsinh_vis_lsb' in cfg.jpg_outputs:
        # unique to GZ, for tidal features etc
        cutout = make_lsb_cutout(vis_im, stretch=20, power=0.5)
        save_image_wrapper(cutout, save_loc.replace('generic', 'gz_arcsinh_vis_lsb'), quality=cfg.jpg_quality)

    ### Space Warps arcinsh processing ###

    if 'sw_arcsinh_vis_only' in cfg.jpg_outputs:
        vis_rgb = m_utils.make_vis_only_cutout(vis_im.copy(), q=500)
        save_image_wrapper(vis_rgb, save_loc.replace('generic', 'sw_arcsinh_vis_only'), quality=cfg.jpg_quality)

    if 'sw_arcsinh_vis_y' in cfg.jpg_outputs:
        vis_y_rgb = make_composite_cutout(vis_im.copy(), y_im.copy(), vis_q=500, nisp_q=1)
        vis_y_rgb_lab = replace_luminosity_channel(vis_y_rgb, rgb_channel_for_luminosity=2, desaturate_speckles=False)
        save_image_wrapper(vis_y_rgb_lab, save_loc.replace('generic', 'sw_arcsinh_vis_y'), quality=cfg.jpg_quality)

    if 'sw_arcsinh_vis_j' in cfg.jpg_outputs:
        vis_j_rgb = make_composite_cutout(vis_im.copy(), j_im.copy(), vis_q=500, nisp_q=1)
        vis_j_rgb_lab = replace_luminosity_channel(vis_j_rgb, rgb_channel_for_luminosity=2, desaturate_speckles=False)
        save_image_wrapper(vis_j_rgb_lab, save_loc.replace('generic', 'sw_arcsinh_vis_j'), quality=cfg.jpg_quality)
    
    if 'sw_arcsinh_vis_y_j' in cfg.jpg_outputs:
        triple_rgb = make_triple_cutout(vis_im.copy(), y_im.copy(), j_im.copy(), short_q=500, mid_q=1, long_q=0.5)
        triple_rgb_lab = replace_luminosity_channel(triple_rgb, rgb_channel_for_luminosity=2, desaturate_speckles=False)
        save_image_wrapper(triple_rgb_lab, save_loc.replace('generic', 'sw_arcsinh_vis_y_j'), quality=cfg.jpg_quality)

    ### Space Warps MTF processing ###

    if any(['mtf' in x for x in cfg.jpg_outputs]):

        vis_mtf = apply_MTF(vis_im)
        # assume if the other bands are available then we will probably want these as well
        if y_im is not None:
            y_mtf = apply_MTF(y_im)
        if j_im is not None:
            j_mtf = apply_MTF(j_im)

        if 'sw_mtf_vis_only' in cfg.jpg_outputs:
            save_image_wrapper(vis_mtf, save_loc.replace('generic', 'sw_mtf_vis_only'), quality=cfg.jpg_quality)

        if 'sw_mtf_vis_y' in cfg.jpg_outputs:
            mean_mtf = np.mean([vis_mtf, y_mtf], axis=0)
            rgb_mtf = np.stack([y_mtf, mean_mtf, vis_mtf], axis=2).astype(np.uint8)
            lab_mtf = replace_luminosity_channel(rgb_mtf, rgb_channel_for_luminosity=2, desaturate_speckles=False)
            save_image_wrapper(lab_mtf, save_loc.replace('generic', 'sw_mtf_vis_y'), quality=cfg.jpg_quality)

        if 'sw_mtf_vis_j' in cfg.jpg_outputs:
            mean_mtf = np.mean([vis_mtf, j_mtf], axis=0)
            rgb_mtf = np.stack([j_mtf, mean_mtf, vis_mtf], axis=2).astype(np.uint8)
            lab_mtf = replace_luminosity_channel(rgb_mtf, rgb_channel_for_luminosity=2, desaturate_speckles=False)
            save_image_wrapper(lab_mtf, save_loc.replace('generic', 'sw_mtf_vis_j'), quality=cfg.jpg_quality)

        if 'sw_mtf_vis_y_j' in cfg.jpg_outputs:
            rgb_mtf = np.stack([j_mtf, y_mtf, vis_mtf], axis=2).astype(np.uint8)
            lab_mtf = replace_luminosity_channel(rgb_mtf, rgb_channel_for_luminosity=2, desaturate_speckles=False)
            save_image_wrapper(lab_mtf, save_loc.replace('generic', 'sw_mtf_vis_y_j'), quality=cfg.jpg_quality)

    logging.debug('Saved all jpg cutouts for single galaxy')

def save_image_wrapper(image, save_loc, quality):
    logging.debug(save_loc)
    subdir = os.path.dirname(save_loc)
    if not os.path.isdir(subdir):
        os.makedirs(subdir, exist_ok=True)
    Image.fromarray(image).save(save_loc, quality=quality)

def make_composite_cutout_from_tiles(source, vis_im, nir_im, allow_radius_estimate=False):
    vis_cutout = m_utils.extract_cutout_from_array(vis_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    nisp_cutout = m_utils.extract_cutout_from_array(nir_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    
    return make_composite_cutout(vis_cutout, nisp_cutout)
    

def make_composite_cutout(vis_cutout, nisp_cutout, vis_q=100, vis_clip=99.85, nisp_q=1, nisp_clip=99.85):
    if vis_cutout.shape != nisp_cutout.shape:
        logging.debug(f'vis shape {vis_cutout.shape}, nisp shape {nisp_cutout.shape}')
        raise AssertionError('Shapes do not match')
    if vis_cutout.size >= 19200**2:
        logging.debug(f'accidentally passed a whole tile, vis size {vis_cutout.size}')
        raise AssertionError('Passed a whole tile')
    if vis_cutout.shape == (19200, 19200):
        logging.debug(f'accidentally passed a whole tile, vis size {vis_cutout.size}')
        raise AssertionError('Passed a whole tile')

    vis_flux_adjusted = m_utils.adjust_dynamic_range(vis_cutout, q=vis_q, clip=vis_clip)
    nisp_flux_adjusted = m_utils.adjust_dynamic_range(nisp_cutout, q=nisp_q, clip=nisp_clip)

    mean_flux = np.mean([vis_flux_adjusted, nisp_flux_adjusted], axis=0)
    
    vis_uint8 = m_utils.to_uint8(vis_flux_adjusted)
    nisp_uint8 = m_utils.to_uint8(nisp_flux_adjusted)
    mean_flux_uint8 = m_utils.to_uint8(mean_flux)

    im = np.stack([nisp_uint8, mean_flux_uint8, vis_uint8], axis=2)
    return im

def make_triple_cutout(short_wav_cutout, mid_wav_cutout, long_wav_cutout, short_q=100, mid_q=.2, long_q=.1, short_clip=99.85, mid_clip=99.85, long_clip=99.85):

    if not short_wav_cutout.shape == mid_wav_cutout.shape == long_wav_cutout.shape:
        logging.debug(f'short shape {short_wav_cutout.shape}, mid shape {mid_wav_cutout.shape}, long shape {long_wav_cutout.shape}')
        raise AssertionError('Shapes do not match')
    if short_wav_cutout.size >= 19200**2:
        logging.debug(f'accidentally passed a whole tile, short wavelength size {short_wav_cutout.size}')
        raise AssertionError('Passed a whole tile')
    if short_wav_cutout.shape == (19200, 19200):
        logging.debug(f'accidentally passed a whole tile, short wavelength size {short_wav_cutout.size}')
        raise AssertionError('Passed a whole tile')

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


    
def MTF_on_normalised_data(x, m):
    """
    Compute the Midtones Transfer Function (MTF) for given x and m.
    This is basically a way to attempt to automatically set the "curves" tool from e.g. photoshop, 
    where m sets the curve and MTF is the application of the curve."""
    # x = np.clip(x, 0, 1)
    assert x.max() <= 1
    assert x.min() >= 0
    y = np.zeros_like(x)
    mask0 = (x == 0)
    maskm = (x == m)
    mask1 = (x == 1)
    mask_else = ~(mask0 | maskm | mask1)
    # these remain fixed
    y[mask0] = 0
    y[maskm] = 0.5
    y[mask1] = 1
    # ..and everything else gets curved
    x_else = x[mask_else]

    # shape of the curve
    numerator = (m - 1) * x_else
    denominator = (2 * m - 1) * x_else - m
    # apply the curve
    y[mask_else] = numerator / denominator
    return y # curved values

def find_m_for_mean(normalized_data, desired_mean, central_crop_size=100):  # 0.1 arcsec per pixel for MER tiles
    # central crop to 100x100
    width = normalized_data.shape[1]
    central_crop_low_edge = width//2 - central_crop_size//2
    central_crop_high_edge = width//2 + central_crop_size//2
    normalized_data_central = normalized_data[central_crop_low_edge:central_crop_high_edge, central_crop_low_edge:central_crop_high_edge]
    x = np.mean(normalized_data_central)
    alpha = desired_mean
    return (x-alpha*x)/(x-2*alpha*x+alpha)

def apply_MTF(image_data, desired_mean_normalized=0.2):  
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize the image data to [0, 1]
    normalized_data = image_data.astype(np.float64)
    min_val = np.min(normalized_data)
    max_val = np.max(normalized_data)
    normalized_data = (normalized_data - min_val) / (max_val - min_val)
    # Desired mean value at 1/5 of the maximum pixel value
    # Raised from 1/8 default from Tian
    # Find the appropriate midtones balance parameter m
    m = find_m_for_mean(normalized_data, desired_mean_normalized)
    # Apply the MTF to the image
    transformed_data = MTF_on_normalised_data(normalized_data, m)
    # Scale transformed data to [0, 255] for JPEG output
    transformed_image_data = (transformed_data * 255).astype(np.uint8)
    return transformed_image_data



    
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
