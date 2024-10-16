"""

Designed for bulk (10k+) download of cutouts and PSF from a list of targets known from external catalogs

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF
"""
import logging
import warnings
import os

from omegaconf import OmegaConf
import pandas as pd
from sklearn.neighbors import KDTree
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.table import Table
from astropy.io.fits.verify import VerifyWarning

from bulk_euclid.utils import pipeline_utils


def run(cfg: OmegaConf):
    logging.info('Starting external targets pipeline')

    create_folders(cfg)
    pipeline_utils.login()

    targets_with_tiles = get_matching_tiles(cfg)  # matching each target with the best tile

    make_cutouts(cfg, targets_with_tiles)

    logging.info('External targets pipeline complete')


def get_matching_tiles(cfg: OmegaConf, external_targets: pd.DataFrame=None):  # simplified from a_make_catalogs_and_cutouts.py
    # create a lookup from external targets
    if external_targets is None:
        external_targets = pd.read_csv(cfg.external_targets_loc)  # must have fields right_ascension (degrees), declination (degrees), field_of_view (arcseconds)

    # all VIS tiles in that release (e.g. F-006 for Wide)
    # we'll get all the other bands and auxillary data later. Here we just need the tile index.
    tiles = pipeline_utils.get_tiles_in_survey(bands=['VIS'], release_name=cfg.release_name)
    assert len(tiles) > 0
    tiles.to_csv('temp_tiles.csv')  # for debugging

    # add tile FoV
    # adds cols of ra_min, ra_max, dec_min, dec_max, from the metadata
    tiles = pipeline_utils.get_tile_extents_fov(tiles)  

    # use KDTree to quickly find the closest tile to each target
    tile_kdtree = KDTree(tiles[['ra', 'dec']].values)  # coordinates of the tile centers
    target_coords = external_targets[['right_ascension', 'declination']].values
    tile_indices = tile_kdtree.query(target_coords, k=1, return_distance=False)
    tile_indices = tile_indices[:, 0]  # the numeric index (0, 1, ...) of the closest tile to each target

    # pick out the closest matching tile for each target
    target_tiles = tiles.iloc[tile_indices].copy()
    # stick together
    target_tiles = pd.concat(target_tiles, external_targets)
    
    # copy over target info (a bit lazy here)
    # target_tiles['id_str'] = external_targets['id_str'].values
    # target_tiles['target_ra'] = external_targets['right_ascension'].values
    # target_tiles['target_dec'] = external_targets['declination'].values
    # target_tiles['target_field_of_view'] = external_targets['field_of_view'].values

    # check if target is within tile FoV
    within_ra = (target_tiles['ra_min'] < target_tiles['target_ra']) & (target_tiles['target_ra'] < target_tiles['ra_max'])
    within_dec = (target_tiles['dec_min'] < target_tiles['target_dec']) & (target_tiles['target_dec'] < target_tiles['dec_max'])
    target_tiles['within_tile'] = within_ra & within_dec
    logging.info(f'Targets within tile FoV: {target_tiles["within_tile"].sum()} of {len(target_tiles)}')

    # filter to only those tiles
    target_tiles = target_tiles[target_tiles['within_tile']]
    assert len(target_tiles) > 0, 'No targets within FoV of any tiles, likely a bug'
    # simplify/explicit for export
    target_tiles  = target_tiles[['tile_index', 'id_str', 'target_ra', 'target_dec', 'target_field_of_view']]
    assert len(target_tiles) > 0

    # target tiles says which tile (index) to use for each target
    return target_tiles


def make_cutouts(cfg: OmegaConf, targets_with_tiles: pd.DataFrame):
    # for each tile index, download all the corresponding data, and make cutouts for each target within that tile
    for tile_index in targets_with_tiles['tile_index'].unique():
        try:
            dict_of_locs = download_all_data_at_tile_index(cfg, tile_index)
            logging.info(f'Downloaded: {dict_of_locs}')
        except AssertionError as e:
            logging.critical(f'Error downloading tile data for {tile_index}')
            logging.critical(e)
            raise e
        
        targets_at_that_index = targets_with_tiles.query(f'tile_index == {tile_index}')

        save_cutouts_for_all_targets_in_that_tile(cfg, tile_index, dict_of_locs, targets_at_that_index)


def download_all_data_at_tile_index(cfg, tile_index):
    flux_tile_metadata = pipeline_utils.get_tiles_in_survey(tile_index=tile_index, bands=cfg.bands, release_name=cfg.release_name)
    # download all the flux tiles with that index
    flux_tile_metadata = pipeline_utils.save_euclid_products(flux_tile_metadata, download_dir=cfg.tile_dir)
    dict_of_locs = flux_tile_metadata[['filter_name', 'file_loc']].set_index('filter_name').to_dict()
    # download all auxillary data for that tile
    for _, flux_tile in flux_tile_metadata.iterrows():
        # could have used tile_index for this search, but we want to restrict to some bands only
        auxillary_tile_metadata = pipeline_utils.get_auxillary_tiles(flux_tile['mosaic_product_oid']) 
        auxillary_tile_metadata = pipeline_utils.save_euclid_products(auxillary_tile_metadata, download_dir=cfg.tile_dir)
        these_aux_locs = auxillary_tile_metadata[['product_type_sas', 'file_loc']].set_index('product_type_sas').to_dict()
        dict_of_locs.update(these_aux_locs)

    return dict_of_locs  # dict of filter_name for bands or product_type_sas of aux tile: file_loc
    # e.g. {
    # 'VIS': 'path/to/VIS.fits',
    # 'NIR_Y': 'path/to/NIR_Y.fits',
    # 'MERPSF': 'path/to/MERPSF.fits',
    # 'MERRMS': 'path/to/MERRMS.fits',
    # 'MERBKG': 'path/to/MERBKG.fits',
    # 'MERFLG': 'path/to/MERFLG.fits'
    # }


def save_cutouts_for_all_targets_in_that_tile(cfg, tile_index, mosaic_product_oid, vis_loc, targets_at_that_index):
    logging.info('loading tile')
    vis_data, vis_header = fits.getdata(vis_loc, header=True)
        # nisp_data = fits.getdata(nisp_y_loc, header=False)
    tile_wcs = WCS(vis_header)
    logging.info('tile loaded')

    # Also extract PSF
    # just VIS at the moment
    psf_loc = pipeline_utils.get_psf_auxillary_tile(mosaic_product_oid, cfg.tile_dir)
    """
        This fits file contains :
        - an image with PSF cutouts of selected objects arranged next to each other. The stamp pixel size can be found in the header keyword STMPSIZE (e.g. 19 for VIS, 33 for NIR).
        - a table giving the match between the PSF cutout center position (columns x_center and y_center) on the PSF grid image and the coordinate in pixels (columns x and y) or on the sky (Ra, Dec) on the MER tile data.
        https://euclid.roe.ac.uk/issues/22495
        """
    psf_tile, psf_header = fits.getdata(psf_loc, ext=1, header=True)
    stamp_size = psf_header['STMPSIZE']
    psf_table = Table.read(fits.open(psf_loc)[2]).to_pandas()
    psf_tree = KDTree(psf_table[['RA', 'Dec']])  # capitals...

    for target_n, target in targets_at_that_index.iterrows():
        logging.info(f'target {target_n} of {len(targets_at_that_index)}')
        target_coord = SkyCoord(target['target_ra'], target['target_dec'], frame='icrs', unit="deg")
        cutout = Cutout2D(data=vis_data, position=target_coord, size=target['target_field_of_view']*u.arcsec, wcs=tile_wcs, mode='partial')

        # find closest matching PSF to target
        _, psf_index = psf_tree.query(target[['target_ra','target_dec']].values.reshape(1, -1), k=1)  # single sample reshape
        # TODO add warning if distance is large (the underscore)
        # scalar: 1 search, with 1 neighbour result
        psf_index = psf_index.squeeze()
        closest_psf = psf_table.iloc[psf_index]
        # this is the metadata row describing the PSF with the closest sky coordinates to the target
        # it has pixel coordinates in the original MER tile (not useful, already handled with sky coordinates) and pixel coordinates in the PSF tile 

        # cutout_psf = Cutout2D(data=psf_tile, position=(closest_psf['RA'], closest_psf['Dec']), size=stamp_size*u.pix)
        cutout_psf = Cutout2D(data=psf_tile, position=(closest_psf['x_center'], closest_psf['y_center']), size=stamp_size)  # no WCS so these are in pixels

        # save both to one FITS file
        save_multifits_cutout(cfg, tile_index, psf_header, target, cutout, cutout_psf)


def save_multifits_cutout(cfg, tile_index, psf_header, target, cutout, cutout_psf):
    hdr = fits.Header()
    hdr.update(cutout.wcs.to_header())  # adds WCS for cutout (vs whole tile)
    header_hdu = fits.PrimaryHDU(header=hdr)
    vis_hdu = fits.ImageHDU(data=cutout.data, name="VIS_FLUX_MICROJANSKY", header=hdr)
    psf_hdu = fits.ImageHDU(data=cutout_psf.data, name="MERPSF", header=psf_header)
    hdul = fits.HDUList([header_hdu, vis_hdu, psf_hdu])

    save_loc = os.path.join(cfg.fits_dir, str(tile_index), str(target['id_str']) + '.fits')
    if not os.path.isdir(os.path.dirname(save_loc)):
        os.mkdir(os.path.dirname(save_loc))
            
    with warnings.catch_warnings():
            # it rewrites my columns to fit the FITS standard by adding HEIRARCH
        warnings.simplefilter('ignore', VerifyWarning)
        hdul.writeto(save_loc, overwrite=True)




def create_folders(cfg: OmegaConf):
    cfg.download_dir = cfg.base_dir + '/' + cfg.name
    cfg.tile_dir = cfg.download_dir + '/tiles'

    cfg.cutout_dir = cfg.download_dir + '/cutouts'
    cfg.fits_dir = cfg.cutout_dir + '/fits'

    cfg.sanity_dir = cfg.download_dir + '/sanity'

    logging.info(f'Saving to {cfg.download_dir}')
    assert os.path.exists(os.path.dirname(cfg.download_dir))
    for d in [
        cfg.download_dir,
        cfg.tile_dir, 
        cfg.cutout_dir,
        cfg.fits_dir,
        cfg.sanity_dir 
        ]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg