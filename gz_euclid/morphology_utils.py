# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: python/Asterism/morphology.py

Created on: 14/09/20
Author: Erik Romelli, Mike Walmsley

Overview
--------
This modules provides the implementation of useful functions for the extraction of
morphometric features
"""

# Standard import
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord


"""
Selection cuts
"""

def apply_zoobot_selection_cuts(df: pd.DataFrame, instrument='VIS'):
    df = add_zoobot_selection_flags(df)
    df['DETAILED_MORPHOLOGY_READY'] = passes_zoobot_selection_cuts(df, instrument)
    return df[df['DETAILED_MORPHOLOGY_READY']]


def passes_zoobot_selection_cuts(dictlike: dict, instrument='VIS'):
    if 'LOW_SEGMENTATION_AREA_VIS_FLAG' not in dictlike.keys():
        dictlike = add_zoobot_selection_flags(dictlike)  # adds all selection cut columns I need
    
    # if instrument == 'VIS':
    #     instrument_size_flag_col = 'LOW_SEGMENTATION_AREA_VIS_FLAG'
    # elif instrument == 'NISP':
    #     instrument_size_flag_col = 'LOW_SEGMENTATION_AREA_NISP_FLAG'

    return ~(dictlike['IS_GAIA_STAR'] | dictlike['BAD_VIS_APER_FLUX_FLAG'] | dictlike['BAD_VIS_APER_FLUX_FLAG'] | dictlike['NO_VIS_DET'] | dictlike['HIGH_SPURIOUS_PROB'] | dictlike['FAILS_FLUX_AND_AREA_CUT'] )


def add_zoobot_selection_flags(df: pd.DataFrame):  # works on either dataframe or dict
    # with dataframe, useful to have these broken out as columns for counting how many sources we lose with each cut

    # no stars
    # MW: updated to also check pd.NAType, which is returned by SAS for final catalog
    df['IS_GAIA_STAR'] = ~(np.isnan(df['GAIA_ID']) | df['GAIA_ID'].isna())

    # flag for bad flux
    df['BAD_VIS_APER_FLUX_FLAG'] = np.isnan(df['FLUX_VIS_APER']) | (df['FLUX_VIS_APER'] < 0)

    df['NO_VIS_DET'] = df['VIS_DET'] != 1
    df['HIGH_SPURIOUS_PROB'] = df['SPURIOUS_PROB'] > 0.2
    df['LOW_SEGMENTATION_AREA'] = df['SEGMENTATION_AREA'] < 0.1
    df['LOW_SEGMENTATION_FLUX'] = df['FLUX_SEGMENTATION'] < 22.90867652  # equiv to VIS mag fainter than 20.5
    df['FAILS_FLUX_AND_AREA_CUT'] = df['LOW_SEGMENTATION_AREA'] & df['LOW_SEGMENTATION_FLUX']


    # df['LOW_SEGMENTED_FLUX_FLAG'] = np.log10(df['SEGMENTATION_AREA']) < 1200
    # df['LOG_SEGMENTED_VIS_MAG_FLAG'] = 

    """
    WHERE flux_vis_aper > 0
    AND gaia_id IS NULL
    AND vis_det=1
    AND spurious_prob < 0.2
    AND (segmentation_area > 1200 OR flux_segmentation > 22.90867652)
    """

    return df


"""
Slicing within MER pipeline
"""

    

def get_cutout_mosaic_coordinates(mosaic, source, buff, allow_radius_estimate=False):
    '''
    Get cutout corners and source centroid for a given source
    '''

    (r, c) = mosaic.shape
    assert r > 0
    assert c > 0
    
    # Get cluster centroid
    cluster_x = source['X_CENTER']
    cluster_y = source['Y_CENTER']

    # r_max is the maximum distance from the source center to the edge of the source segmap, in pixels
    # segmap is calculated by asterism in MER_DEBLENDING
    # r_max is in the deblending catalog but NOT the final catalog, so it needs to be estimated outside of MER_Morphology
    try:
        source_r_max = source['R_MAX']
    except KeyError:
        if allow_radius_estimate:
            source_r_max = estimate_source_r_max(source)
        else:
            raise KeyError(f'Source radius not found and radius estimation is not allowed: {source}')
    assert source_r_max > 0, f'source_r_max {source_r_max} < 0'
    # try to slice a square cutout
    # can still ultimately be non-square if galaxy is near edge (via min, max below)
    # thanks to tile overlap, these are hopefully covered by the other tile
    dy = source_r_max + buff
    dx = source_r_max + buff

    # calculate cutout corners
    # do not return corners off the low edge (i.e. below 0)
    # return corners 
    r1 = np.max([0, cluster_y-dy]).astype(int)
    r2 = np.min([r, cluster_y+dy]).astype(int)
    c1 = np.max([0, cluster_x-dx]).astype(int)  # pixel index of lower left corner of cutout
    c2 = np.min([c, cluster_x+dx]).astype(int)

    # sanity check; if you attempt to make cutouts from the wrong tile, this will fail
    assert r1 < r2, f'cutout low y edge above high y edge: (r1, r2)={(r1, r2)}, cluster_y={cluster_y}, mosaic={mosaic.shape}'
    assert c1 < c2, f'cutout low x edge above high x edge: (c1, c2)={(c1, c2)}, cluster_x={cluster_x}, mosaic={mosaic.shape}'

    # sanity check: not massive
    assert r2 - r1 < 10000, f'Cutout too large: {r2 - r1}'
    assert c2 - c1 < 10000, f'Cutout too large: {c2 - c1}'

    # sanity check: non-negative
    assert r2 - r1 > 0, f'Cutout edges inverted: {r1, r2}'
    assert c2 - c1 > 0, f'Cutout edges inverted: {c1, c2}'

    # calculate cutout center
    # MW: isn't this only halfway between cluster centroid and cutout edge?
    x_center = cluster_x - c1
    y_center = cluster_y - r1
    
    return x_center, y_center, r1, r2, c1, c2


def estimate_source_r_max(source):
    intercept = -0.35048866
    coefs = {'LOG_SEGMENTATION_AREA': 0.506900163942416, 'ELLIPTICITY': 0.2405883433225405, 'LOG_KRON_RADIUS': 0.11148176647655159}
    log_r_max_estimate = intercept + source['LOG_SEGMENTATION_AREA'] * coefs['LOG_SEGMENTATION_AREA'] + source['ELLIPTICITY'] * coefs['ELLIPTICITY'] + source['LOG_KRON_RADIUS'] * coefs['LOG_KRON_RADIUS']
    return 10 ** log_r_max_estimate


def extract_cutout_from_array(mosaic: np.ndarray, source, buff, allow_radius_estimate=False, enforce_shape=True):
    # pixel coordinates of cutout (r=x, c=y, not sure why this syntax)
    _, _, r1, r2, c1, c2 = get_cutout_mosaic_coordinates(
        mosaic,
        source,
        buff,
        allow_radius_estimate
    )
    # cut mosaic
    data = np.copy(mosaic[r1:r2+1, c1:c2+1])
    if enforce_shape:
        assert np.abs(data.shape[0] - data.shape[1]) <= 1, f'Shape mismatch: cutout of shape {data.shape} not square'
        # sometimes it is 1 pixel off, probably from rounding, and that's okay
    return data




"""
Image processing
"""

def adjust_dynamic_range(flux, q=100, clip=99.85):
    im = np.arcsinh(flux * q)
    if clip < 100:
        im = np.clip(im, 0, np.percentile(im, clip))
    return  im


def to_uint8(im, clip_below_zero=True):
    if clip_below_zero:
        im = np.clip(im, 0, None)
    im = (im + im.min()) / (im.min() + im.max())
    return (255 * im).astype(np.uint8)


def fits_to_pandas(fits_loc):
    '''
    Convert a fits table to a pandas dataframe, taking care of <NA> values
    '''
    final_cat = Table.read(fits_loc).to_pandas()
    # final_cat = final_cat.replace('<NA>', np.nan)  # doesn't work <NA> is a NaT and doesn't seem to be replaced
    final_cat = final_cat.replace({np.nan: None}).replace({None: np.nan})  # dumb but works. NaT counts as nan, becomes None along with normal nan, and then all get set back to nan
    return final_cat




def make_vis_only_cutout_from_tiles(source: dict, vis_im, allow_radius_estimate=False):
    vis_cutout = extract_cutout_from_array(vis_im, source, buff=0, allow_radius_estimate=allow_radius_estimate)
    return make_vis_only_cutout(vis_cutout)

def make_vis_only_cutout(vis_cutout):
    vis_flux_adjusted = adjust_dynamic_range(vis_cutout, q=100, clip=99.85)
    vis_uint8 = to_uint8(vis_flux_adjusted)
    return vis_uint8





def convert_final_cat_to_within_pipeline_cat(df, vis_tile_loc):
    # Zoobot cutout code is designed to expect the columns of the intermediate catalog
    # but the final catalog does not include these columns
    # so we need to do some footwork to reconstruct(ish) the intermediate catalog columns

    # inside the pipeline, we would have the R_MAX column
    # mock this by estimating R_MAX
    df['LOG_SEGMENTATION_AREA'] = np.log10(df['SEGMENTATION_AREA'])
    df['LOG_KRON_RADIUS'] = np.log10(df['KRON_RADIUS'])
    df['R_MAX_ESTIMATED'] = df.apply(estimate_source_r_max, axis=1)
    df['R_MAX'] = df['R_MAX_ESTIMATED']
    # clean up the new columns
    del df['R_MAX_ESTIMATED']
    del df['LOG_SEGMENTATION_AREA']
    del df['LOG_KRON_RADIUS']

    # simlarly, as final cat does not include these
    vis_pixel_centers = get_pixel_centers(df, vis_tile_loc)
    df['X_CENTER'] = vis_pixel_centers[0]
    df['Y_CENTER'] = vis_pixel_centers[1]

    # similarly (this is just used for validation)
    df['SOURCE_ID'] = df['OBJECT_ID']

    return df

def get_pixel_centers(df, fits_loc):
    wcs_frame = load_wcs_from_file(fits_loc)
    world_c = SkyCoord(df["RIGHT_ASCENSION"], df["DECLINATION"], frame='icrs', unit="deg")
    pix_c = wcs_frame.world_to_pixel(world_c)
    return pix_c


def load_wcs_from_file(filename):
    # https://docs.astropy.org/en/stable/wcs/loading_from_fits.html
    # Load the FITS hdulist using astropy.io.fits
    hdulist = fits.open(filename)
    # Parse the WCS keywords in the primary HDU
    w = wcs.WCS(hdulist[0].header)
    return w