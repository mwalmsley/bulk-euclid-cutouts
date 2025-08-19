import os
import logging
import glob
    
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.table import Table


from bulk_euclid.utils import pipeline_utils


def run(cfg):
    cfg = pipeline_utils.create_folders(cfg)

    tiles = pipeline_utils.find_available_tiles(cfg)

    logging.info(f'Tiles to make cutouts from: {len(tiles)}')

    tiles = sorted(tiles, key=lambda x: x.tile_index)  # sort by tile index, so that the order is deterministic

    while len(tiles) > 0:
        tile = tiles.pop()

        logging.info(f'tile {tile.tile_index}, plus {len(tiles)} left')
        try:
            make_volunteer_cutouts(cfg, tile)
        except AssertionError as e:
            logging.warning('Skipping tile {} due to fatal error'.format(tile.tile_index))
            logging.warning(e)

        del tile  # free memory explicitly

    logging.info('Cutout creation complete')
    



def tile_passes_filters(tile, cfg):
    """
    Check if a tile passes the filters set in the configuration.
    """
    # Check if the tile has all mosaics for all required bands
    for band in cfg.bands:
        band_mosaics = tile[band]
        if band_mosaics is None: 
            logging.warning(f'Tile {tile["tile_index"]} is missing band {band}')
            return False
        
        # no need to check all required data products, already done in find_available_tiles
        # if any(band_mosaics.empty:
        #     logging.warning(f'Tile {tile["tile_index"]} has empty mosaics for band {band}')
        #     return False

    
    # hardcoded: remove a few bad tiles which currently have very little data in Q1
    if cfg.release_name == 'Q1_R1':
        logging.info('Removing bad tiles for Q1_R1')
        bad_tile_indices = [102018211, 102160873, 102021021]
        if tile.tile_index in bad_tile_indices:
            logging.warning(f'Tile {tile["tile_index"]} is in the list of bad tiles, skipping')
            return False

    return True


def add_cutout_paths(cfg, catalog):
    # will be used like .jpg -> output_name.jpg later
    if cfg.jpg_outputs:
        # e.g. jpg_loc/generic/102159774/102159774_123456_generic.jpg
        catalog['jpg_loc_generic'] = catalog.apply(
            lambda x: pipeline_utils.get_cutout_loc(cfg.jpg_dir, x, output_format='jpg', version_suffix='generic', oneway_hash=False), axis=1)
   
    if cfg.fits_outputs:  # true or false, unlike jpg_loc:
        catalog['fits_loc'] = catalog.apply(
            lambda x: pipeline_utils.get_cutout_loc(cfg.fits_dir, x, output_format='fits.gz', version_suffix=None, oneway_hash=False), axis=1)


def make_volunteer_cutouts(cfg: OmegaConf, tile: pipeline_utils.Tile):

    logging.info(f'Tile {tile.tile_index}')
    tile_catalog_loc = cfg.catalog_dir + f'/{tile.tile_index}_mer_catalog.csv'

    if (not os.path.isfile(tile_catalog_loc)) or cfg.refresh_catalogs:

        all_tile_sources = Table.read(tile.mer_final_catalog).to_pandas()
        # all columns are upper, make lower
        all_tile_sources.columns = all_tile_sources.columns.str.lower()
        relevant_tile_sources = pipeline_utils.find_relevant_sources_in_tile(cfg, df=all_tile_sources)
        if relevant_tile_sources.empty:
            logging.warning(f'Tile {tile.tile_index} has no relevant sources, skipping cutouts')
            return
        logging.info(relevant_tile_sources[['right_ascension', 'declination']].mean())
        add_cutout_paths(cfg, relevant_tile_sources)  # add save locs here, useful later
        relevant_tile_sources.to_csv(tile_catalog_loc, index=False)

    else:
        logging.info(f'Catalog already exists at {tile_catalog_loc}, loading')
        relevant_tile_sources = pd.read_csv(tile_catalog_loc)
        if relevant_tile_sources.empty:
            logging.warning(f'Tile {tile.tile_index} has no relevant sources, skipping cutouts')
            return

    add_cutout_paths(cfg, relevant_tile_sources)  # update save locs just in case
    pipeline_utils.save_cutouts(cfg, tile, relevant_tile_sources)







    # RA and Dec of tile are not actually used, only here for sanity check - could touch to open with WCS, but easier to drop
    # assert not tiles.duplicated(subset=['ra', 'dec', 'instrument_name', 'filter_name']).any()

    # # visual sanity check
    # plt.scatter(tiles['ra'], tiles['dec'], s=2., color='r', label='Tile centers')
    # plt.xlabel('Right Ascension')
    # plt.ylabel('Declination')
    # plt.legend()
    # # unlike the tiles, which are in SAS (albeit wrongly indexed), the MER catalogs are only available in SAS for a small corner of the Wide survey
    # plt.savefig(cfg.sanity_dir + '/tile_centers.png')

    # return tiles


# def select_tiles(cfg, tiles) -> pd.DataFrame:

#     # tiles needs columns: ['tile_index', 'filter_name', 'file_name', 'release_name', 'mosaic_product_oid']

#     rng = np.random.default_rng(cfg.seed)

#     # filter name will only include the cfg.bands, due to the query in get_tiles_in_survey
#     is_missing_bands = tiles.pivot(index='tile_index', columns='filter_name', values='file_name').isna().any(axis=1) # series like {tile_index: is_missing_bands}. file_name not used.
#     possible_indices = is_missing_bands[~is_missing_bands].index  # flip to get indices with all bands, then get index
#     logging.info(f'Num. of tiles with all bands: {len(possible_indices)}')

#     if cfg.num_tiles > 0:
#         logging.info(f'Randomly subselecting {cfg.num_tiles} tiles')
#         assert len(possible_indices) > cfg.num_tiles, f'Not enough tiles with both VIS and Y: {len(possible_indices)}'
#         tile_indices_to_use = rng.choice(possible_indices, cfg.num_tiles, replace=False)
#         logging.info(f'Num. of tiles to use after random subselection: {len(tile_indices_to_use)}')
#     else:
#         logging.info('Using all tiles')
#         tile_indices_to_use = possible_indices

#     tiles_to_use = tiles[tiles['tile_index'].isin(tile_indices_to_use)].reset_index(drop=True) 
#     assert len(tiles_to_use) == len(cfg.bands) * len(tile_indices_to_use), f'{len(tiles_to_use)} != len(cfg.bands) ({len(cfg.bands)}) * {len(tile_indices_to_use)}'

#     return tiles_to_use



# def download_tile_and_catalog(cfg, tiles_to_download: pd.DataFrame, tile_index: int):
#     # tiles_to_download is a df of all tiles, including metadata like datalabs_path if available

#     # if cfg.download_method == 'sas':
#     #     # df of paths to downloaded tiles, keyed by 'file_loc'
#     #     downloaded_tiles = pipeline_utils.download_mosaics(tile_index, tiles_to_download, cfg.tile_dir)
#     # else:
#     assert cfg.download_method == 'datalabs_path'
#     downloaded_tiles = tiles_to_download.query(f'tile_index == {tile_index}').copy()
#     # instead of downloading, just point the path to datalabs
#     downloaded_tiles['file_loc'] = downloaded_tiles['datalabs_path'] + '/' + downloaded_tiles['file_name']  # slightly counterintuitive, datalabs_path is the directory only
#     logging.info(f'Tile locations: {downloaded_tiles["file_loc"] }')

#     tile_metadata_to_copy = dict()  # scalars
#     tile_metadata_to_copy['tile_index'] = tile_index
#     # use VIS for RA, Dec, release name. Only one release name allowed so should all be the same or very similar.
#     vis_tile = downloaded_tiles.query('filter_name == "VIS"').iloc[0]
#     tile_metadata_to_copy['ra'] = vis_tile['ra']
#     tile_metadata_to_copy['dec'] = vis_tile['dec'] 
#     tile_metadata_to_copy['release_name'] = vis_tile['release_name']
#     # record the tile file locations for each band
#     for band in cfg.bands:
#         tile_metadata_to_copy[f'{band.lower()}_loc'] = downloaded_tiles.query(f'filter_name == "{band}"')['file_loc'].squeeze()

#         if cfg.add_bkg:
#             mosaic_product_oid = downloaded_tiles.query(f'tile_index == {tile_index} & filter_name == "{band}"')['mosaic_product_oid']
#             bkg_tiles = pipeline_utils.get_auxillary_tiles(mosaic_product_oid.iloc[0], auxillary_products = ['MERBKG'])
#             bkg_tile_loc = bkg_tiles.iloc[0]['datalabs_path'] + '/' + bkg_tiles.iloc[0]['file_name']
#             tile_metadata_to_copy[f'{band.lower()}_bkg_loc'] = bkg_tile_loc

#     tile_catalog = get_and_save_tile_catalog(cfg, tile_index, tile_metadata_to_copy)
#     return tile_catalog




# pretty much cannot locally debug, requires datalabs
# if __name__ == "__main__":

#     cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/v2_challenge_launch.yaml')

    # run(cfg)