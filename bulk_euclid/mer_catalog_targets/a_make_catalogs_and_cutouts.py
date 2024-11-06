import os
import logging
# from tqdm.notebook import tqdm # TODO ask Kristin to add?

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bulk_euclid.utils import pipeline_utils


def run(cfg):
    pipeline_utils.login(cfg)
    cfg = create_folders(cfg)
    tiles = get_tile_catalog(cfg)
    tiles = select_tiles(cfg, tiles)
    

    print(tiles.columns.values())
    print(tiles.head())
    print(tiles['release_name'].value_counts)
    exit()

    for tile_n, tile_index in enumerate(tiles['tile_index'].unique()):
        logging.info(f'tile {tile_index}: {tile_n} of {len(tiles)}')
        try:
            tile_catalog = download_tile_and_catalog(cfg, tiles, tile_index)

            make_volunteer_cutouts(cfg, tile_catalog)
            if cfg.delete_tiles:
                logging.info('Deleting tile')
                vis_loc = tile_catalog['vis_tile'].iloc[0]
                nisp_loc = tile_catalog['y_tile'].iloc[0]
                os.remove(vis_loc)
                os.remove(nisp_loc)
        except AssertionError as e:
            logging.warning('Skipping tile {} due to fatal error'.format(tile_index))
            logging.warning(e)

    logging.info('Cutout creation complete')
    

def create_folders(cfg: OmegaConf):
    cfg.download_dir = cfg.base_dir + '/' + cfg.name
    cfg.tile_dir = cfg.download_dir + '/tiles'
    cfg.catalog_dir = cfg.download_dir + '/catalogs'

    cfg.cutout_dir = cfg.download_dir + '/cutouts'
    cfg.jpg_dir = cfg.cutout_dir + '/jpg'
    cfg.fits_dir = cfg.cutout_dir + '/fits'

    cfg.sanity_dir = cfg.download_dir + '/sanity'

    logging.info(f'Saving to {cfg.download_dir}')
    assert os.path.exists(os.path.dirname(cfg.download_dir))
    for d in [
        cfg.download_dir,
        cfg.tile_dir, 
        cfg.catalog_dir,
        cfg.jpg_dir,
        cfg.sanity_dir 
        ]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg


def get_tile_catalog(cfg: OmegaConf):
    # currently only south and wide have any data

    # see pipeline_utils
    # survey = pipeline_utils.WIDE

    tiles = pipeline_utils.get_tiles_in_survey(bands=cfg.bands, release_name=cfg.release_name)  # F-003_240321 recently appeared
    
    logging.info(tiles['instrument_name'].value_counts())
    logging.info(tiles['release_name'].value_counts())
    assert not tiles.duplicated(subset=['ra', 'dec', 'instrument_name', 'filter_name']).any()

    # logging.info(f'Tiles after restricting to southern area: {len(tiles)}')

    # visual sanity check
    plt.scatter(tiles['ra'], tiles['dec'], s=2., color='r', label='Tile centers')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.legend()
    # unlike the tiles, which are in SAS (albeit wrongly indexed), the MER catalogs are only available in SAS for a small corner of the Wide survey
    plt.savefig(cfg.sanity_dir + '/tile_centers.png')

    return tiles


def select_tiles(cfg, tiles):
    rng = np.random.default_rng(cfg.seed)

    # tiles.groupby('tile_index')['filter_name'].unique().value_counts()
    # filter name will only include the cfg.bands, due to the query in get_tiles_in_survey
    is_missing_bands = tiles.pivot(index='tile_index', columns='filter_name', values='file_name').isna().any(axis=1) # series like {tile_index: is_missing_bands}. file_name not used.
    possible_indices = is_missing_bands[~is_missing_bands].index  # flip to get indices with all bands, then get index
    logging.info(f'Num. of tiles with all bands: {len(possible_indices)}')

    # vis_tiles = tiles.query('filter_name == "VIS"')
    # y_tiles = tiles.query('filter_name == "NIR_Y"')
    # possible_indices = list(set(vis_tiles['tile_index']).intersection(set(y_tiles['tile_index'])))
    # logging.info(f'Num. of tiles with all required bands ({cfg.bands}): {len(possible_indices)}')

    if cfg.num_tiles > 0:
        logging.info(f'Randomly subselecting {cfg.num_tiles} tiles')
        assert len(possible_indices) > cfg.num_tiles, f'Not enough tiles with both VIS and Y: {len(possible_indices)}'
        tile_indices_to_use = rng.choice(possible_indices, cfg.num_tiles, replace=False)
        logging.info(f'Num. of tiles to use after random subselection: {len(tile_indices_to_use)}')
    else:
        logging.info('Using all tiles')
        tile_indices_to_use = possible_indices

    tiles_to_use = tiles[tiles['tile_index'].isin(tile_indices_to_use)].reset_index(drop=True) 
    assert len(tiles_to_use) == len(cfg.bands) * len(tile_indices_to_use), f'{len(tiles_to_use)} != len(cfg.bands) ({len(cfg.bands)}) * {len(tile_indices_to_use)}'

    return tiles_to_use


def download_tile_and_catalog(cfg, tiles_to_download: pd.DataFrame, tile_index: int):
    download_tiles = pipeline_utils.download_mosaics(tile_index, tiles_to_download, cfg.tile_dir)
    # vis_tile = download_tiles.query('filter_name == "VIS"')['file_loc'].squeeze()
    # nisp_tile = download_tiles.query('filter_name == "NIR_Y"')['file_loc'].squeeze()
    # vis_loc = vis_tile['file_loc']
    # nisp_loc = nisp_tile['file_loc']

    tile_metadata_to_copy = dict()  # scalars
    tile_metadata_to_copy['tile_index'] = tile_index
    # use VIS for RA, Dec, release name. Only one release name allowed so should all be the same or very similar.
    vis_tile = download_tiles.query('filter_name == "VIS"').iloc[0]
    tile_metadata_to_copy['ra'] = vis_tile['ra']
    tile_metadata_to_copy['dec'] = vis_tile['dec'] 
    tile_metadata_to_copy['release_name'] = vis_tile['release_name']
    # record the tile file locations for each band
    for band in cfg.bands:
        tile_metadata_to_copy[f'{band.lower()}_loc'] = download_tiles.query(f'filter_name == "{band}"')['file_loc'].squeeze()

    logging.debug(tile_metadata_to_copy)

    tile_catalog = get_and_save_tile_catalog(cfg, tile_index, tile_metadata_to_copy)
    return tile_catalog


def get_and_save_tile_catalog(cfg, tile_index: int, tile_metadata_to_copy: dict) -> pd.DataFrame:
    # tile_index = vis_tile['tile_index']
    tile_catalog_loc = cfg.catalog_dir + f'/{tile_index}_mer_catalog.csv'
    if (not os.path.isfile(tile_catalog_loc)) or cfg.refresh_catalogs:
        tile_galaxies = pipeline_utils.find_relevant_sources_in_tile(cfg, tile_index)
        assert not tile_galaxies.empty
  
        tile_galaxies['tile_index_from_segmentation_map_id'] = tile_galaxies['segmentation_map_id'].apply(lambda x: int( str(x)[:9] ))  # first 9 digits are tile index
        logging.info(tile_galaxies['tile_index_from_segmentation_map_id'].value_counts())

        # add metadata
        for key, value in tile_metadata_to_copy.items():
            tile_galaxies[key] = value

        add_cutout_paths(cfg, tile_galaxies)  # inplace
        tile_galaxies.to_csv(tile_catalog_loc, index=False)
    else:
        logging.info(f'Catalog already exists at {tile_catalog_loc}, loading')
        tile_galaxies = pd.read_csv(tile_catalog_loc)
    return tile_galaxies


def add_cutout_paths(cfg, catalog):
    # will be used like .jpg -> output_name.jpg later
    if cfg.jpg_outputs:
        # e.g. jpg_loc/generic/102159774/102159774_123456_generic.jpg
        catalog['jpg_loc_generic'] = catalog.apply(
            lambda x: pipeline_utils.get_cutout_loc(cfg.jpg_dir, x, output_format='jpg', version_suffix='generic', oneway_hash=False), axis=1)
   
    if cfg.fits_outputs:  # true or false, unlike jpg_loc:
        catalog['fits_loc'] = catalog.apply(
            lambda x: pipeline_utils.get_cutout_loc(cfg.fits_dir, x, output_format='fits.gz', version_suffix=None, oneway_hash=False), axis=1)


def make_volunteer_cutouts(cfg, df):
    valid_tile_indices = list(df['tile_index'].unique())
    logging.info(f'Tiles to make cutouts from: {len(valid_tile_indices)}')

    for tile_n, tile_index in enumerate(valid_tile_indices):
        logging.info(f'Tile {tile_index}, {tile_n}')
        tile_galaxies = df.query(f'tile_index == {tile_index}')
        logging.info(tile_galaxies[['right_ascension', 'declination']].mean())
        pipeline_utils.save_cutouts(cfg, tile_galaxies)


# pretty much cannot locally debug, requires Euclid data access
# if __name__ == "__main__":

#     cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/v2_challenge_launch.yaml')

    # run(cfg)
