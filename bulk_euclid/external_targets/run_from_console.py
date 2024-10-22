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

    # some ad hoc setup

    external_targets = pd.read_csv(cfg.external_targets_loc)
    external_targets = external_targets.rename(columns={'ra': 'target_ra', 'dec': 'target_dec', 'ID': 'id_str'})
    del external_targets['Unnamed: 0']
    external_targets['target_field_of_view'] = 20  # arcseconds
    # TODO Karina to remove these duplicates
    external_targets = external_targets.drop_duplicates(subset=['id_str'], keep='first')

    pipeline.run(cfg)

    logging.info('Done :)')

    

if __name__ == "__main__":

    cfg = OmegaConf.load('configs/external_targets_latest.yaml')

    cfg.log_file = cfg.base_dir + f'/external_targets_{cfg.name}_{time.time()}.log'

    logging.basicConfig(
        level=logging.INFO, filename=cfg.log_file, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run(cfg)
