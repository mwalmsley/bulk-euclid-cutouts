import time
import logging

from omegaconf import OmegaConf

import a_download_tiles_and_catalogs
import b_create_master_catalog_and_cutouts

def run(cfg):
    a_download_tiles_and_catalogs.run(cfg)
    b_create_master_catalog_and_cutouts.run(cfg)

    

if __name__ == "__main__":

    cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/v3_challenge_midaug.yaml')

    cfg.log_file = cfg.base_dir + f'/pipeline_{time.time()}.log'

    logging.basicConfig(level=logging.INFO, filename=cfg.log_file, filemode='w')

    run(cfg)
