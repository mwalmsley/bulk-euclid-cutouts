import os
import glob
import json
import random
import datetime
import logging

import numpy as np
from shared_astro_utils import subject_utils, upload_utils, time_utils
import tqdm
import hashlib
from PIL import Image
from panoptes_client import Project
import pandas as pd


IM_COLS = ['jpg_loc_composite', 'jpg_loc_vis_only', 'jpg_loc_vis_lsb']


def resize_image(current_loc, new_loc, size, overwrite=False):

    if os.path.exists(new_loc) and not overwrite:
         return
    
    if not os.path.exists(os.path.dirname(new_loc)):
        os.makedirs(os.path.dirname(new_loc))
    
    im = Image.open(current_loc)
    # good moment to check image doesn't have missing pixels/bands
    if im.mode == 'RGB':
        arr = np.array(im)
        per_band_max = arr.max(axis=(0,1))
        # print(per_band_max)
        if per_band_max.min() == 0:  # axis 2 is the bands
            logging.warning('Image has missing pixels, skipping: ' + current_loc)
            return
        if np.isnan(arr).any():
            logging.warning('Image has nan pixels, skipping: ' + current_loc)
            return
        if (arr == 0).mean() > 0.15:
            logging.warning('Image has more than 15% zero pixels, skipping: ' + current_loc)
            return
    im.resize(size, resample=Image.Resampling.LANCZOS).save(new_loc)


def get_hash(id_str, extra_key):
        str_to_hash = id_str + extra_key
        return hashlib.sha256(str_to_hash.encode()).hexdigest()


def get_id_str(df):
     return df['release_name'] + '_' + df['tile_index'].astype(int).astype(str) + '_' + df['object_id'].astype(int).astype(str).str.replace('-', 'NEG')


def get_resized_loc(galaxy, im_col, save_dir):
    # places in {save_dir}/{tile_subdir}/{original filename}_424.jpg
    tile_subdir = os.path.basename(os.path.dirname(galaxy[im_col]))
    new_filename = os.path.basename(galaxy[im_col]).replace('.jpg', '_424.jpg')
    return os.path.join(save_dir, tile_subdir, new_filename)


def select_galaxies(version_name):
    df_loc = f'/home/walml/repos/gz-euclid-datalab/data/pipeline/{version_name}/catalogs/_master_catalog.csv'

    df = pd.read_csv(df_loc)

    # apply the final filter for galaxies good to classify but not to upload
    # for now, anything about 25k pixels
    df = df[df['segmentation_area'] < 25000]
    # and anything very small (and bright)
    df = df[df['segmentation_area'] > 200]

    # and do not upload anything either in the overlap regions or with a catalog older than the overlap region calculation
    # NOW REMOVED as overlap region was impossible to calculate accurately
    # print(df['in_tile_overlap_region'].value_counts())
    # print(df['in_tile_overlap_region'].isna().sum())
    # exit()
    
    # instead, only upload galaxies in the core areas of the tiles (as selected by MER)
    # only objects in tile core areas have a segmentation_map_id/are in the MER final catalog
    # calculated by parsing out tile_index from within segmentation_map_id (first 9 digits)
    # NO LONGER calculated using kdtree to pick the closest tile in the tiling strategy table
    df = df[df['this_tile_index_is_best'] == True]
    print(len(df))

    # HISTORICAL note
    # I specifically picked these tiles for the first upload, based on the tiles with maximum distance to nearest tile
    # launch_tiles = [
    #     102015620, 102021061, 102016036, 102021034, 102015615, 102034406,
    #    102012400, 102013966, 102026083, 102011655, 102027664, 102033849,
    #    102020090, 102023521, 102018234, 102019150, 102027661, 102016463,
    #    102022002, 102030421, 102021511, 102031525, 102026603, 102030405,
    #    102022988, 102016054, 102018712, 102022015, 102022017, 102021057,
    #    102022027, 102032104, 102028219, 102028213, 102034444, 102032115,
    #    102022990, 102031550, 102032117, 102022013, 102036817, 102018254,
    #    102025018, 102023993, 102027667, 102028753, 102029879, 102030997,
    #    102026063, 102035627
    # ]

    # use saved catalogs from previous uploads to check what has been uploaded already
    # for now, only run uploads from local desktop
    previous_upload_cat_locs = glob.glob('/home/walml/repos/gz-euclid-datalab/data/pipeline/zooniverse_upload/*.csv')
    if len(previous_upload_cat_locs) == 0:
        logging.warning('No previous uploads found')
        previous_uploads = pd.DataFrame()
    else:
        previous_uploads = pd.concat([pd.read_csv(loc) for loc in previous_upload_cat_locs])
        already_uploaded_tile_indices_from_exports = list(previous_uploads['tile_index'].unique())
        tiles_to_avoid = set(already_uploaded_tile_indices_from_exports)
        # don't upload those
        df = df[~df['tile_index'].isin(tiles_to_avoid)].reset_index(drop=True)
    logging.info('Galaxies to upload, after final filters and avoiding previous tiles: ' + str(len(df)))
    return df


def run_resizing(version_name, df, overwrite=False):
    logging.info(f'Galaxies to resize: {len(df)}')

    local_data_dir = f'/home/walml/repos/gz-euclid-datalab/data/pipeline/{version_name}'
    resized_dir = local_data_dir + '/resized'

    # print(df['object_id'].value_counts())
    assert df['object_id'].value_counts().max() == 1, 'There are duplicate object ids'

    # fix paths to original cutouts, and prepare paths for resized images
    for col in IM_COLS:
        # adjust for local upload
        # e.g. {version_name}/cutouts/{tile_subdir}/{original filename}
        df[col] = df[col].apply(lambda x: f'{local_data_dir}/cutouts/{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}')  
        logging.info('Checking paths for original and resized cutouts: ' + df.iloc[0][col])
        # keep tile subdir
        df[col + '_ready_to_resize'] = df[col].apply(lambda x: os.path.isfile(x))
        assert df[col + '_ready_to_resize'].any(), 'No cutouts ready to resize, check paths e.g. ' + df.iloc[0][col]
        df[col + '_resized'] = df.apply(lambda x: get_resized_loc(x, col, resized_dir), axis=1)
        # print(df[col][0])
        logging.info('Paths ready for original and resized cutouts: ' + col)
        
    all_images_ready = df[[col + '_ready_to_resize' for col in IM_COLS]].all(axis=1)
    logging.info(f'Images ready: {all_images_ready.sum()} of {len(df)}')
    # drop all images not ready to be resized
    df = df[all_images_ready].reset_index(drop=True)
    assert len(df) > 0

    logging.info(f'Resizing images to {resized_dir}')
    for _, subject in tqdm.tqdm(df.iterrows(), total=len(df), unit='resized'):
        for im_col in IM_COLS:
            current_loc = subject[im_col]
            new_loc = subject[im_col + '_resized']  # create new cols for the new locations
            # print(current_loc)
            assert current_loc != new_loc, 'You are about to overwrite the original image'
            resize_image(current_loc, new_loc, (424, 424), overwrite=overwrite)

    # some resizes will have been skipped so check again
    for col in IM_COLS:
        df[col + '_resized_successfully'] = df[col + '_resized'].apply(lambda x: os.path.isfile(x))
    df = df[df[[col + '_resized_successfully' for col in IM_COLS]].all(axis=1)].reset_index(drop=True)
    logging.info(f'Images resized successfully: {len(df)}')

    return df


def run_upload_by_tiles(df, tiles_to_upload, debug=False):
    df = df.copy()
    assert not df.empty

    # for galaxies in those tiles, completely shuffle df, will now be uploading random galaxies in random order (from tileset b and those selected tiles only)
    df = df[df['tile_index'].isin(tiles_to_upload)]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info('For upload, galaxies per tile: ')
    logging.info(df['tile_index'].value_counts())
    assert not df.empty

    # add metadata
    df['id_str'] = get_id_str(df)
    df['!filename'] = df['id_str'].apply(lambda x: get_hash(x, extra_key='_racoon'))
    # make manifest
    df['locations'] = df[[col + '_resized' for col in IM_COLS]].apply(lambda x: list(x), axis=1)
    df['#campaign'] = 'euclid_challenge'
    df['#upload_date'] = time_utils.current_date()
    df['metadata'] = df[['!filename', '#campaign', '#upload_date']].to_dict(orient='records')   # column of dicts, confusingly


    if debug:
        # for testing   
        subject_set_name = f'{datetime.datetime.now().strftime("%Y_%m_%d")}_euclid_challenge_tileset_b_dev'
    else:
        subject_set_name = f'{datetime.datetime.now().strftime("%Y_%m_%d")}_euclid_challenge_tileset_b_tiles_{tile_low}_{tile_high}'
        df.to_csv(f'/home/walml/repos/gz-euclid-datalab/data/pipeline/zooniverse_upload/master_catalog_during_{subject_set_name}.csv', index=False)
    # exit()

    # fast async way
    # manifest = df_to_upload[['locations', 'metadata']].to_dict(orient='records')
    # print(len(manifest))
    # print(manifest[0])
    # upload_utils.bulk_upload_subjects(subject_set_name, manifest, project_id='5733', async_batch_size=1)


    # slower way

    subject_utils.authenticate()

    project = Project.find(5733)

    for _, subject in tqdm.tqdm(df.iterrows(), total=len(df), unit='subjects'):
        locations = list(subject[[col + '_resized' for col in IM_COLS]])
        metadata = subject[['!filename', '#campaign', '#upload_date']]
        subject_utils.upload_subject(locations, project, subject_set_name, metadata)

        # https://github.com/zooniverse/kade/pull/170/files



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, force=True)

    version_name = 'v3_challenge_tileset_b'

    df = select_galaxies(version_name)
    # tileset b will be the tiles for the second block of uploads after launch (the first being launch tiles)
    tileset_b = df['tile_index'].unique().tolist()
    # now frozen to avoid accidental changes later
    # json.dump(tileset_b, open('/home/walml/repos/euclid-morphology/upload/tileset_b.json', 'w'))
    # record it here for later adding to 'tiles to avoid'

    df = run_resizing(version_name, df, overwrite=False)
    # exit()

    # subject_set_name = '2024_07_31_euclid_challenge_first_50_tiles_0_19900'
    # df = df[:19900]
    # subject_set_name = '2024_07_31_euclid_challenge_first_50_tiles_19900_end'
    # df = df[19900:]
    # print(df['id_str'])
    # print(df['!filename'])
    # exit()

    # df has 130k entries, and I would prefer to upload tile-by-tile
    # so let's pick 1 tile for each 1k galaxies

    tileset_b = json.load(open('/home/walml/repos/euclid-morphology/upload/tileset_b.json'))
    # shuffle
    np.random.default_rng(42).shuffle(tileset_b)

    # tile_low = 0
    # tile_high = 4
    # tile_low = 4
    # tile_high = 6
    # tile_low = 6
    # tile_high = 10
    # tile_low = 10
    # tile_high = 15
    # tile_low = 15
    # tile_high = 20
    # tile_low = 20
    # tile_high = 30
    # tile_low = 30
    # tile_high = 50
    # tile_low = 50
    # tile_high = 70
    # tile_low = 70
    # tile_high = 100
    # tile_low = 100
    # tile_high = 150
    # TODO last tile block
    tile_low = 150
    tile_high = 999
    tiles_to_upload = tileset_b[tile_low:tile_high]
    # print(tiles_to_upload)

    run_upload_by_tiles(df, tiles_to_upload, debug=False)
