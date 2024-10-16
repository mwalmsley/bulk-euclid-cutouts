import time
import logging
import os
from omegaconf import OmegaConf


def run(cfg):

    # copied from pipeline/run_pipeline.py

    # temp hack to "install" this package, pending a proper dockerfile
    # does not seem to pip install -e correctly
    import sys
    repo_dir = '/media/user/repos/bulk-euclid-cutouts'
    os.path.isdir(repo_dir)
    sys.path.insert(0,repo_dir)

    from bulk_euclid.external_targets import a_run
    logging.info('Import successful')

    a_run.run(cfg)

    logging.info('Done :)')

    

if __name__ == "__main__":

    cfg = OmegaConf.load('configs/external_targets_debug.yaml')

    cfg.log_file = cfg.base_dir + f'/external_targets_{cfg.name}_{time.time()}.log'

    logging.basicConfig(
        level=logging.INFO, filename=cfg.log_file, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run(cfg)
