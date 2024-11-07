import glob
import logging
import shutil

import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

from bulk_euclid.utils import pipeline_utils


def run(cfg):
    master_catalog = group_to_master_catalog(cfg)
    # master_catalog = pd.read_csv(cfg.catalog_dir + '/_master_catalog.csv')

    visualise_catalog(cfg, master_catalog)

    zip_for_download(cfg)



def group_to_master_catalog(cfg: OmegaConf):

    # quick check
    catalog_locs = glob.glob(cfg.catalog_dir + '/*_mer_catalog.csv')
    logging.info(f'Found {len(catalog_locs)} catalogs')
    master_catalog = pd.concat([pd.read_csv(c) for c in catalog_locs], axis=0)
    logging.info(f'Found {len(master_catalog)} sources')
    
    # add_cutout_paths(cfg, master_catalog)
                                                            
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
    save_loc = cfg.sanity_dir + '/segmentation_area_vs_vis_mag.png'
    logging.info(f'Saving catalog viz to {save_loc}')
    plt.savefig(save_loc)
    return fig, ax



def zip_for_download(cfg: OmegaConf):
    logging.info('Zipping cutouts and catalogs')
    # save to e.g. v1_challenge_launch_cutouts.zip
    shutil.make_archive(cfg.download_dir + '_catalogs', 'zip', root_dir=cfg.catalog_dir)
    logging.info('Zipped catalogs')
    if cfg.jpg_outputs:
        for output_format in cfg.jpg_outputs:
            shutil.make_archive(cfg.cutout_dir + f'_jpg_cutouts_{output_format}', 'zip', root_dir=cfg.jpg_dir + '/' + output_format)  # zips from: download/cutouts/jpg/output_format
            logging.info(f'Zipped {output_format} jpg cutouts')
        logging.info('Zipped all jpg cutouts')
    if cfg.fits_outputs:
        shutil.make_archive(cfg.cutout_dir + '_fits_cutouts', 'zip', root_dir=cfg.fits_dir)
        logging.info('Zipped fits cutouts')



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
    



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # cfg = OmegaConf.load('configs/v3_challenge_midaug.yaml')
    cfg = OmegaConf.load('configs/local_debug.yaml')
    
    run(cfg)