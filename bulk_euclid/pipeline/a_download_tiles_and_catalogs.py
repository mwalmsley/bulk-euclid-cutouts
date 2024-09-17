import os
import logging
# from tqdm.notebook import tqdm # TODO ask Kristin to add?

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bulk_euclid.utils import pipeline_utils


def run(cfg):
    login()
    cfg = create_folders(cfg)
    tiles = get_tile_catalog(cfg)
    tiles = select_tiles(tiles)
    download_tiles(cfg, tiles, refresh_catalogs=False)

def login():

    if os.path.isdir('/media/home/team_workspaces'):
        from astroquery.esa.euclid.core import Euclid
        # two line file, username and password
        # do not commit or put in any team workspace, obviously...
        Euclid.login(credentials_file='/media/home/_credentials/euclid_login.txt')
    else:
        raise ValueError('Not on DataLabs')


def create_folders(cfg: OmegaConf):
    cfg.download_dir = cfg.base_dir + '/' + cfg.name
    cfg.tile_dir = cfg.download_dir + '/tiles'
    cfg.catalog_dir = cfg.download_dir + '/catalogs'
    cfg.cutout_dir = cfg.download_dir + '/cutouts'
    cfg.jpg_dir = cfg.cutout_dir + '/jpg'
    cfg.sanity_dir = cfg.cutout_dir + '/sanity'

    logging.info(f'Saving to {cfg.download_dir}')
    assert os.path.exists(cfg.download_dir)

    for d in [cfg.jpg_dir, cfg.tile_dir, cfg.catalog_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg


def get_tile_catalog(cfg: OmegaConf):
    # currently only south and wide have any data

    # see pipeline_utils
    survey = pipeline_utils.WIDE

    tiles = pipeline_utils.get_tiles_in_survey(survey, bands=['VIS', 'NIR_Y'], release_name=cfg.release_name)  # F-003_240321 recently appeared
    
    logging.info(tiles['instrument_name'].value_counts())
    logging.info(tiles['release_name'].value_counts())
    assert not tiles.duplicated(subset=['ra', 'dec', 'instrument_name']).any()

    # visual sanity check
    plt.scatter(tiles['ra'], tiles['dec'], s=2., color='r', label='Tile centers')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.legend()
    # unlike the tiles, which are in SAS (albeit wrongly indexed), the MER catalogs are only available in SAS for a small corner of the Wide survey
    plt.savefig(cfg.sanity_dir + '/tile_centers.png')

    tiles = tiles.query(f'ra < {cfg.ra_upper_limit}').query(f'dec < {cfg.dec_upper_limit}').reset_index(drop=True)
    logging.info(f'Tiles after restricting to southern area: {len(tiles)}')
    # TODO automate/remove this hack

    # add tile extents (previously useful for querying the MER catalog, but now no longer used)
    # tiles = pipeline_utils.get_tile_extents_fov(tiles)

    return tiles


def select_tiles(cfg, tiles):
    rng = np.random.default_rng(cfg.seed)

    vis_tiles = tiles.query('filter_name == "VIS"')
    y_tiles = tiles.query('filter_name == "NIR_Y"')
    possible_indices = list(set(vis_tiles['tile_index']).intersection(set(y_tiles['tile_index'])))
    logging.info(f'Num. of tiles with VIS and Y: {len(possible_indices)}')

    assert len(possible_indices) > cfg.num_tiles, f'Not enough tiles with both VIS and Y: {len(possible_indices)}'
    tile_indices_to_use = rng.choice(possible_indices, cfg.num_tiles, replace=False)
    tiles_to_use = tiles[tiles['tile_index'].isin(tile_indices_to_use)].reset_index(drop=True)  
    logging.info(f'Num. of tiles to use after random subselection: {len(tiles_to_use)}')
    # should be exactly twice as many tiles to use as sampled (1 for vis, 1 for y)
    assert len(tiles_to_use) == 2 * cfg.num_tiles
    return tiles_to_use

def download_tiles(cfg: OmegaConf, tiles_to_download, refresh_catalogs=False):

    for tile_n, tile_index in enumerate(tiles_to_download['tile_index'].unique()): 
        
        logging.info(f'{tile_n} of {len(tiles_to_download)}', tile_index)

        vis_loc, nisp_loc = pipeline_utils.download_mosaics(tile_index, tiles_to_download, cfg.tile_dir)
        try:
            vis_tile = tiles_to_download.query('filter_name == "VIS"').query(f'tile_index == {tile_index}').squeeze()
            tile_catalog_loc = cfg.catalog_dir + f'/{tile_index}_mer_catalog.csv'
            if (not os.path.isfile(tile_catalog_loc)) or refresh_catalogs:
                tile_galaxies = pipeline_utils.find_zoobot_sources_in_tile(vis_tile)
                assert not tile_galaxies.empty
  
                tile_galaxies['tile_index_from_segmentation_map_id'] = tile_galaxies['segmentation_map_id'].apply(lambda x: int( str(x)[:9] ))  # first 9 digits are tile index
                logging.info(tile_galaxies['tile_index_from_segmentation_map_id'].value_counts())
                tile_galaxies['this_tile_index_is_best'] = tile_galaxies['tile_index_from_segmentation_map_id'] == tile_index

                tile_galaxies['vis_tile'] = vis_loc
                tile_galaxies['y_tile'] = nisp_loc
                tile_galaxies['tile_ra'] = vis_tile['ra']
                tile_galaxies['tile_dec'] = vis_tile['dec']
                tile_galaxies['release_name'] = vis_tile['release_name']
                tile_galaxies.to_csv(tile_catalog_loc, index=False)
        except AssertionError as e:
            logging.critical(e)
    logging.info('All downloads complete, safe to time out from Euclid auth')


# pretty much cannot locally debug, requires Euclid data access
# if __name__ == "__main__":

#     cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/v2_challenge_launch.yaml')

    # run(cfg)
