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

    from bulk_euclid.mer_catalog_targets import a_make_catalogs_and_cutouts, b_export
    logging.info('Import successful')

    a_make_catalogs_and_cutouts.run(cfg)
    b_export.run(cfg)

    logging.info('Done :)')

    

if __name__ == "__main__":

    cfg = OmegaConf.load('configs/mer_catalog_targets/mer_catalog_random_tiles_q1.yaml')
    # cfg = OmegaConf.load('configs/v4_post_euclid_challenge.yaml')

    cfg.log_file = cfg.base_dir + f'/pipeline_{cfg.name}_{time.time()}.log'


    # logging.basicConfig(
    #     level=logging.DEBUG, 
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )

    logging.basicConfig(
        level=logging.INFO, filename=cfg.log_file, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # also log to console

    run(cfg)
