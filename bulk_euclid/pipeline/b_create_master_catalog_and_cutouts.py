import glob
import logging
import shutil

import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

from bulk_euclid.utils import pipeline_utils


def run(cfg):
    make_galaxy_catalog(cfg)
    master_catalog = pd.read_csv(cfg.catalog_dir + '/_master_catalog.csv')
    visualise_catalog(cfg, master_catalog)
    make_volunteer_cutouts(master_catalog)
    # make_fits_cutouts(master_catalog)


def make_galaxy_catalog(cfg: OmegaConf):

    # quick check
    catalog_locs = glob.glob(cfg.catalog_dir + '/*_mer_catalog.csv')
    logging.info(f'Found {len(catalog_locs)} catalogs')
    master_catalog = pd.concat([pd.read_csv(c) for c in catalog_locs], axis=0)
    logging.info(f'Found {len(master_catalog)} sources')

    # TODO apply any galaxy-level random subselection?
    # nah, I think tile-level subselection will be enough. It's clearer to do every relevant source in a given tile. Then we have a list of classified tiles.

    master_catalog['jpg_loc_composite'] = master_catalog.apply(
        lambda x: pipeline_utils.get_cutout_loc(cfg.jpg_dir, x, output_format='jpg', version_suffix='composite', oneway_hash=False), axis=1)
    master_catalog['jpg_loc_vis_only'] = master_catalog.apply(
        lambda x: pipeline_utils.get_cutout_loc(cfg.jpg_dir, x, output_format='jpg', version_suffix='vis_only', oneway_hash=False), axis=1)
    master_catalog['jpg_loc_vis_lsb'] = master_catalog.apply(
        lambda x: pipeline_utils.get_cutout_loc(cfg.jpg_dir, x, output_format='jpg', version_suffix='vis_lsb', oneway_hash=False), axis=1)
                                                            

    master_catalog.to_csv(cfg.catalog_dir + '/_master_catalog.csv', index=False)  # _ to appear first
    return master_catalog.reset_index(drop=True)


def visualise_catalog(cfg: OmegaConf, df):

    fig, ax = plt.subplots()
    ax.scatter(df['mag_segmentation'], df['segmentation_area'], alpha=.1, s=1.)
    ax.set_yscale('log')
    plt.xlabel('VIS Mag')
    plt.ylabel('Segmentation area (VIS pixels)')
    x_min = 18
    x_max = 22
    y_min = 10**2
    y_max = 10**5
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.hlines(1200, 20.5, x_max, linestyle='-', alpha=.7, color='r', label='Area=1200px')
    ax.vlines(20.5, y_min, 1200, linestyle='-', alpha=.7, color='g', label='VIS=20.5')

    alpha = .05
    color = 'r'
    select_x = np.linspace(x_min, x_max)

    select_x = np.linspace(20.5, x_max)
    ax.fill_between(select_x, y_min, 1200, color=color, alpha=alpha)
    # ax.text(19.1, 400, 'Complete to VIS=20.5')
    # ax.text(20.1, 5000, 'Includes faint extended galaxies')

    plt.legend(loc='upper right')
    save_loc = cfg.catalog_dir + '/segmentation_area_vs_vis_mag.png'
    logging.info(f'Saving catalog viz to {save_loc}')
    plt.savefig(save_loc)
    return fig, ax


def make_volunteer_cutouts(df):
    valid_tile_indices = list(df['tile_index'].unique())
    logging.info(f'Tiles to make cutouts from: {len(valid_tile_indices)}')

    for tile_n, tile_index in enumerate(valid_tile_indices):
        logging.info(f'Tile {tile_index}, {tile_n}')
        tile_galaxies = df.query(f'tile_index == {tile_index}')
        logging.info(tile_galaxies[['right_ascension', 'declination']].mean())
        vis_loc = tile_galaxies['vis_tile'].iloc[0]
        nisp_loc = tile_galaxies['y_tile'].iloc[0]
        pipeline_utils.save_cutouts(vis_loc, nisp_loc, tile_galaxies, overwrite=False)


# def make_fits_cutouts(df):
#     # also save the cutouts as FITS, for other people to tinker with 
#     # much more space-expensive, but still a lot smaller than the original tiles

#     valid_tile_indices = [102010567]
#     # valid_tile_indices = list(df['tile_index'].unique())
#     for tile_n, tile_index in enumerate(valid_tile_indices):
#         logging.info(f'Tile {tile_index}, {tile_n}')
#         tile_galaxies = df.query(f'tile_index == {tile_index}')[:10]
#         vis_loc = tile_galaxies['vis_tile'].iloc[0]
#         nisp_loc = tile_galaxies['y_tile'].iloc[0]
#         pipeline_utils.save_cutouts(vis_loc, nisp_loc, tile_galaxies, overwrite=True, output_format='fits.gz')
    

def zip_for_download(cfg: OmegaConf):
    # save to e.g. v1_challenge_launch_cutouts.zip
    # shutil.make_archive(cfg.download_dir + '_cutouts', 'zip', root_dir=cfg.cutout_dir)  # should be jpg only
    shutil.make_archive(cfg.download_dir + '_catalogs', 'zip', cfg.catalog_dir)
    logging.info('Zipped cutouts and catalogs')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # cfg = OmegaConf.load('configs/v3_challenge_midaug.yaml')
    cfg = OmegaConf.load('configs/local_debug.yaml')
    
    run(cfg)