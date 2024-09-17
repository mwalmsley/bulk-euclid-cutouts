from bulk_euclid.utils import morphology_utils_ou_mer as m_utils  # try to keep this exactly like ou mer version

import numpy as np

# import Cutout2d
from astropy.nddata.utils import Cutout2D
# import WCS
from astropy.wcs import WCS



def make_composite_cutout_from_tiles(source, vis_im, nir_im, allow_radius_estimate=False):
    vis_cutout = m_utils.extract_cutout_from_array(vis_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    nisp_cutout = m_utils.extract_cutout_from_array(nir_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    
    return make_composite_cutout(vis_cutout, nisp_cutout)
    
def make_composite_cutout(vis_cutout, nisp_cutout):
    # print('starting composite')

    assert vis_cutout.shape == nisp_cutout.shape, f'vis shape {vis_cutout.shape}, nisp shape {nisp_cutout.shape}'
    assert vis_cutout.size < 19200**2, f'accidentally passed a whole tile, vis size {vis_cutout.size}'
    assert vis_cutout.shape != (19200, 19200), f'accidentally passed a whole tile, vis size {vis_cutout.size}'
    # print(vis_cutout.shape)
    # print(vis_cutout.min(), vis_cutout.max())


    # print('test complete')
    vis_flux_adjusted = m_utils.adjust_dynamic_range(vis_cutout, q=100, clip=99.85)
    # print('vis_flux_adjusted ready')
    nisp_flux_adjusted = m_utils.adjust_dynamic_range(nisp_cutout, q=.2, clip=99.85)

    mean_flux = np.mean([vis_flux_adjusted, nisp_flux_adjusted], axis=0)
    # print('mean ready')
    
    vis_uint8 = m_utils.to_uint8(vis_flux_adjusted)
    nisp_uint8 = m_utils.to_uint8(nisp_flux_adjusted)
    mean_flux_uint8 = m_utils.to_uint8(mean_flux)
    # print('uint8 ready')

    im = np.stack([nisp_uint8, mean_flux_uint8, vis_uint8], axis=2)
    # print('stack ready')
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
