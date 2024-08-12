import os
import logging
# from tqdm.notebook import tqdm # TODO ask Kristin to add?

from omegaconf import OmegaConf
import astropy.units as u
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from astropy.table import Table


from gz_euclid import pipeline_utils



def create_folders(cfg: OmegaConf):
    cfg.tile_dir = cfg.download_dir + '/tiles'
    cfg.catalog_dir = cfg.download_dir + '/catalogs'
    cfg.cutout_dir = cfg.download_dir + '/cutouts'
    cfg.jpg_dir = cfg.cutout_dir + '/jpg'

    logging.info(f'Saving to {cfg.download_dir}')
    assert os.path.exists(cfg.download_dir)

    for d in [cfg.jpg_dir, cfg.tile_dir, cfg.catalog_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg


def get_tile_catalog(cfg: OmegaConf):
    # currently only south and wide have any data

    # see pipeline_utils
    # survey = pipeline_utils.EDF_PV
    survey = pipeline_utils.WIDE

    tiles = pipeline_utils.get_tiles_in_survey(survey, bands=['VIS', 'NIR_Y'], release_name=cfg.release_name)  # F-003_240321 recently appeared
    # tiles['tile_index']

    logging.info(tiles['instrument_name'].value_counts())
    logging.info(tiles['release_name'].value_counts())

    vis_tiles = tiles.query('instrument_name == "VIS"').reset_index(drop=True)
    assert not vis_tiles.duplicated(subset=['ra', 'dec']).any()

    # # visual sanity check
    # plt.scatter(tiles['ra'], tiles['dec'], s=2., color='r', label='Tile centers')
    # plt.xlabel('Right Ascension')
    # plt.ylabel('Declination')
    # plt.legend()
    # # unlike the tiles, which are in SAS (albeit wrongly indexed), the MER catalogs are only available in SAS for a small corner of the Wide survey
    # plt.savefig('tile_centers.png')

    tiles = tiles.query(f'ra < {cfg.ra_upper_limit}').query(f'dec < {cfg.dec_upper_limit}').reset_index(drop=True)
    logging.info(f'Tiles after restricting to southern area: {len(tiles)}')

    # add tile extents (useful for querying the MER catalog)
    tiles = pipeline_utils.get_tile_extents_fov(tiles)
    return tiles


def select_tiles(cfg, tiles):
    rng = np.random.default_rng(cfg.seed)

    vis_tiles = tiles.query('filter_name == "VIS"')
    y_tiles = tiles.query('filter_name == "NIR_Y"')
    possible_indices = list(set(vis_tiles['tile_index']).intersection(set(y_tiles['tile_index'])))
    logging.info(f'Num. of tiles with VIS and Y: {len(possible_indices)}')

    # tile_indices_to_use = rng.choice(possible_indices, 100)
    # [102034406, 102033246, 102012403,
    assert len(possible_indices) > cfg.num_tiles, f'Not enough tiles with both VIS and Y: {len(possible_indices)}'
    tile_indices_to_use = rng.choice(possible_indices, cfg.num_tiles, replace=False)
    tiles_to_use = tiles[tiles['tile_index'].isin(tile_indices_to_use)].reset_index(drop=True)  
    logging.info(f'Num. of tiles to use after random subselection: {len(tiles_to_use)}')
    # should be exactly twice as many tiles to use as sampled (1 for vis, 1 for y)
    assert len(tiles_to_use) == 2 * cfg.num_tiles
    return tiles_to_use

# class TileFinder():

#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.kdtree, self.tile_indices = make_tile_kdtree(cfg)
#         logging.info('tile strategy kdtree ready')

#     def get_tile_index_of_closest_tile(self, ra, dec):
#         integer_index = self.kdtree.query([[ra, dec]], k=1, return_distance=False)[0]  # [[]] as searching for one row only
#         integer_index_of_closest = integer_index[0]
#         return self.tile_indices[integer_index_of_closest]
    
# def make_tile_kdtree(cfg):
#     tiling_strategy = Table.read(cfg.tiling_strategy_loc).to_pandas()  # '/home/walml/repos/gz-euclid-datalab/data/tiling_plan/field_all_sky_overview.fits'
#     all_tile_coords = tiling_strategy[['RA', 'Dec']].values  # includes both normal and special, so will only make cutouts if no closer special tile
#     kdtree = KDTree(all_tile_coords)
#     tile_indices = tiling_strategy['tileId'].values  # tile_index is tileId here
#     return kdtree, tile_indices


def download_tiles(cfg: OmegaConf, tiles_to_download, refresh_catalogs=False):

    # tile_finder = TileFinder(cfg)

    for tile_n, tile_index in enumerate(tiles_to_download['tile_index'].unique()): 
        
        logging.info(f'{tile_n} of {len(tiles_to_download)}', tile_index)

        vis_loc, nisp_loc = pipeline_utils.download_mosaics(tile_index, tiles_to_download, cfg.tile_dir)
        try:
            vis_tile = tiles_to_download.query('filter_name == "VIS"').query(f'tile_index == {tile_index}').squeeze()
            tile_catalog_loc = cfg.catalog_dir + f'/{tile_index}_mer_catalog.csv'
            if (not os.path.isfile(tile_catalog_loc)) or refresh_catalogs:
                tile_galaxies = pipeline_utils.find_zoobot_sources_in_tile(vis_tile)
                assert not tile_galaxies.empty
                # tile_galaxies['tile_index_of_closest_tile'] = tile_galaxies.apply(lambda x: tile_finder.get_tile_index_of_closest_tile(x['right_ascension'], x['declination']), axis=1)
                # # tile_galaxies['this_tile_index_is_best'] = tile_galaxies['tile_index_of_closest_tile'].apply(lambda x: x == tile_index)
                # print(tile_index)
                # tile_galaxies['this_tile_index_is_best'] = tile_galaxies['tile_index_of_closest_tile'] == tile_index
                # print(tile_galaxies['tile_index_of_closest_tile'].value_counts())
                tile_galaxies['tile_index_from_segmentation_map_id'] = tile_galaxies['segmentation_map_id'].apply(lambda x: int( str(x)[:9] ))  # first 9 digits are tile index
                print(tile_galaxies['tile_index_from_segmentation_map_id'].value_counts())
                tile_galaxies['this_tile_index_is_best'] = tile_galaxies['tile_index_from_segmentation_map_id'] == tile_index

                tile_galaxies['vis_tile'] = vis_loc
                tile_galaxies['y_tile'] = nisp_loc
                tile_galaxies['tile_ra'] = vis_tile['ra']
                tile_galaxies['tile_dec'] = vis_tile['dec']
                tile_galaxies['release_name'] = vis_tile['release_name']
                tile_galaxies.to_csv(tile_catalog_loc, index=False)
        except AssertionError as e:
            print(e)
    print('All downloads complete, safe to time out from Euclid auth')



# if __name__ == "__main__":

#     cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/v2_challenge_launch.yaml')

#     cfg = create_folders(cfg)
#     tiles = get_tile_catalog(cfg)
#     tiles = select_tiles(tiles)
#     download_tiles(cfg, tiles, refresh_catalogs=False)
