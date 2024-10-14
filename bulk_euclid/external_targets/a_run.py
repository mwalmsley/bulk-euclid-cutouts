"""

Designed for bulk (10k+) download of cutouts and PSF from a list of targets known from external catalogs

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF
"""
import logging
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

from bulk_euclid.utils import pipeline_utils


def run(cfg: OmegaConf):
    logging.info('Starting external targets pipeline')

    create_folders(cfg)
    pipeline_utils.login()

    tiles, target_tiles = get_matching_tiles(cfg)

    print(tiles.head())
    print(target_tiles.head())
    print(tiles['tile_index'].value_counts())
    print((tiles['tile_index'] == 929).sum())
    print(target_tiles['tile_index'].value_counts())

    make_cutouts(cfg, tiles, target_tiles)

    logging.info('External targets pipeline complete')


def get_matching_tiles(cfg: OmegaConf):  # simplified from a_make_catalogs_and_cutouts.py

    external_targets = pd.read_csv(cfg.external_targets_loc)  # must have fields right_ascension (degrees), declination (degrees), field_of_view (arcseconds)

    tiles = pipeline_utils.get_tiles_in_survey(bands=['VIS', 'NIR_Y'], release_name=cfg.release_name)
    assert len(tiles) > 0

    vis_tiles = tiles.query('filter_name == "VIS"')

    # add tile FoV
    vis_tiles = pipeline_utils.get_tile_extents_fov(vis_tiles)  # ra_min, ra_max, dec_min, dec_max, approximate

    # use KDTree to quickly find closest tile to each target
    # print(vis_tiles.columns.values)
    tile_kdtree = KDTree(vis_tiles[['ra', 'dec']].values)  # coordinates of the tile centers
    target_coords = external_targets[['right_ascension', 'declination']].values
    tile_indices = tile_kdtree.query(target_coords, k=1, return_distance=False)
    print(tile_indices.shape)
    tile_indices = tile_indices[:, 0]
    print(tile_indices.shape)

    # pick out the closest matching tile for each target
    target_tiles = vis_tiles.iloc[tile_indices]
    # target_tiles['tile_index'] = target_tiles.index
    # target_tiles['tile_index'] = target_tiles['tile_index'].astype(int)

    # copy over target info (a bit lazy here)
    target_tiles['target_ra'] = external_targets['right_ascension'].values
    target_tiles['target_dec'] = external_targets['declination'].values
    target_tiles['target_field_of_view'] = external_targets['field_of_view'].values

    # check if target is within tile FoV
    within_ra = (target_tiles['ra_min'] < target_tiles['target_ra']) & (target_tiles['target_ra'] < target_tiles['ra_max'])
    within_dec = (target_tiles['dec_min'] < target_tiles['target_dec']) & (target_tiles['target_dec'] < target_tiles['dec_max'])
    target_tiles['within_tile'] = within_ra & within_dec
    logging.info(f'Targets within tile FoV: {target_tiles["within_tile"].sum()} of {len(target_tiles)}')

    # filter to only those tiles
    target_tiles = target_tiles[target_tiles['within_tile']]
    assert len(target_tiles) > 0, 'No targets within FoV of any tiles, likely a bug'
    # simplify/explicit for export
    target_tiles  = target_tiles[['tile_index', 'target_ra', 'target_dec', 'target_field_of_view']]
    assert len(target_tiles) > 0

    # tiles is all tiles valid for download (from all bands)
    # target tiles says which tile (index) to use for each target
    return tiles, target_tiles


def make_cutouts(cfg: OmegaConf, tiles, target_tiles):
    for tile_index in target_tiles['tile_index'].unique():
        try:
            downloaded_tiles = pipeline_utils.download_mosaics(tile_index, tiles, download_dir=cfg.tile_dir)
            vis_tile = downloaded_tiles.query('filter_name == "VIS"').squeeze()
            vis_loc = vis_tile['file_loc']
            print(vis_loc)
        except AssertionError as e:
            logging.critical(e)
            raise e
        targets_at_that_index = target_tiles.query(f'tile_index == {tile_index}')

        logging.info('loading tile')
        vis_data, header = fits.getdata(vis_loc, header=True)
        # nisp_data = fits.getdata(nisp_y_loc, header=False)
        tile_wcs = WCS(header)
        logging.info('tile loaded')

        # Also extract PSF
        # just VIS at the moment
        psf_loc = pipeline_utils.get_psf_auxillary_tile(vis_tile, cfg.fits_dir)
        """
        This fits file contains :
        - an image with PSF cutouts of selected objects arranged next to each other. The stamp pixel size can be found in the header keyword STMPSIZE (e.g. 19 for VIS, 33 for NIR).
        - a table giving the match between the PSF cutout center position (columns x_center and y_center) on the PSF grid image and the coordinate in pixels (columns x and y) or on the sky (Ra, Dec) on the MER tile data.
        https://euclid.roe.ac.uk/issues/22495
        """
        psf_tile, header = fits.getdata(psf_loc, ext=1, header=True)
        stamp_size = header['STMPSIZE']
        psf_table = Table.read(fits.open(psf_loc)[2]).to_pandas()
        psf_tree = KDTree(psf_table[['RA', 'Dec']])  # capitals...

        for target_n, target in targets_at_that_index.iterrows():
            logging.info(f'target {target_n} of {len(targets_at_that_index)}')
            target_coord = SkyCoord(target['target_ra'], target['target_dec'], frame='icrs', unit="deg")
            cutout = Cutout2D(data=vis_data, position=target_coord, size=target['field_of_view']*u.arsec, wcs=tile_wcs, mode='partial')
            # TODO save vis cutout

            # find closest matching PSF to target
            _, psf_index = psf_tree.query([target['target_ra'], target['target_dec']])
            # TODO add warning if distance is large (the underscore)
            closest_psf = psf_table.iloc[psf_index]
            cutout_psf = Cutout2D(data=psf_tile, position=(closest_psf['RA'], closest_psf['Dec']), size=stamp_size*u.pix)
            # TODO save PSF


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