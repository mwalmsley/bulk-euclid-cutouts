import time
import logging
import os

import pandas as pd
from omegaconf import OmegaConf


def run(cfg):

    # copied from pipeline/run_pipeline.py

    # temp hack to "install" this package, pending a proper dockerfile
    # does not seem to pip install -e correctly
    import sys
    repo_dir = '/media/user/repos/bulk-euclid-cutouts'
    os.path.isdir(repo_dir)
    sys.path.insert(0,repo_dir)

    from bulk_euclid.external_targets import pipeline
    logging.info('Import successful')

    pipeline.run(cfg)

    logging.info('Done :)')

    

if __name__ == "__main__":

    # config_loc = 'configs/external_targets/external_targets_master_list.yaml'
    # config_loc = 'configs/external_targets/q1_lenses.yaml'  # targets from discovery engine etc
    config_loc = 'configs/strong_lensing_candidates_outside_discovery_engine_q1_v1.yaml'   # targets from outside engine
    cfg = OmegaConf.load(config_loc)

    cfg.log_file = cfg.base_dir + f'/{cfg.name}_{time.time()}.log'

    logging.basicConfig(
        level=logging.INFO, filename=cfg.log_file, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # also log to console

    # logging.basicConfig(
    #     level=logging.DEBUG, 
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )

    run(cfg)
