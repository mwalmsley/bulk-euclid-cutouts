import time
import logging
import os
from omegaconf import OmegaConf


def run(cfg):

    # temp hack to "install" this package, pending a proper dockerfile
    # does not seem to pip install -e correctly
    import sys
    repo_dir = '/media/user/repos/bulk-euclid-cutouts'
    os.path.isdir(repo_dir)
    sys.path.insert(0,repo_dir)

    from bulk_euclid.pipeline import a_download_tiles_and_catalogs, b_create_master_catalog_and_cutouts
    logging.info('Success')
    exit()

    a_download_tiles_and_catalogs.run(cfg)
    b_create_master_catalog_and_cutouts.run(cfg)

    

if __name__ == "__main__":

    cfg = OmegaConf.load('/home/walml/repos/gz-euclid-datalab/run_pipeline/remote_debug.yaml')

    cfg.log_file = cfg.base_dir + f'/pipeline_{time.time()}.log'

    logging.basicConfig(level=logging.INFO, filename=cfg.log_file, filemode='w')

    run(cfg)
