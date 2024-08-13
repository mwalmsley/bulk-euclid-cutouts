import os
import glob
import json
import random
import datetime

import numpy as np
from shared_astro_utils import subject_utils, upload_utils, time_utils
import tqdm
import hashlib
from PIL import Image
# from panoptes_client import Project
import pandas as pd


def resize_image(current_loc, new_loc, size, overwrite=False):

    if os.path.exists(new_loc) and not overwrite:
         return
    
    if not os.path.exists(os.path.dirname(new_loc)):
        os.makedirs(os.path.dirname(new_loc))
    
    Image.open(current_loc).resize(size, resample=Image.Resampling.LANCZOS).save(new_loc)


def get_hash(id_str, extra_key):
        str_to_hash = id_str + extra_key
        return hashlib.sha256(str_to_hash.encode()).hexdigest()


def get_id_str(df):
     return df['release_name'] + '_' + df['tile_index'].astype(int).astype(str) + '_' + df['object_id'].astype(int).astype(str).str.replace('-', 'NEG')


def get_resized_loc(galaxy, im_col):
    tile_subdir = os.path.basename(os.path.dirname(galaxy[im_col]))
    new_filename = os.path.basename(galaxy[im_col]).replace('.jpg', '_424.jpg')
    return os.path.join(save_dir, tile_subdir, new_filename)


if __name__ == '__main__':


    # df_loc = '/home/walml/repos/euclid-morphology/datalabs/data/pipeline/v2_challenge_launch_local/catalogs/_master_catalog.csv'

    version_name = 'v3_challenge_midaug'
    df_loc = f'/home/walml/repos/gz-euclid-datalab/data/pipeline/{version_name}/catalogs/_master_catalog.csv'
    local_data_dir = f'/home/walml/repos/gz-euclid-datalab/data/pipeline/{version_name}'


    df = pd.read_csv(df_loc)

    # apply the final filter for galaxies good to classify but not to upload
    # for now, anything about 25k pixels
    df = df[df['segmentation_area'] < 25000]
    # and anything very small (and bright)
    df = df[df['segmentation_area'] > 200]

    # and do not upload anything either in the overlap regions or with a catalog older than the overlap region calculation
    # print(df['in_tile_overlap_region'].value_counts())
    # print(df['in_tile_overlap_region'].isna().sum())
    # exit()

    # NO LONGER calculated using kdtree to pick the closest tile in the tiling strategy table
    # now calculated by parsing out tile_index from within segmentation_map_id (first 9 digits)
    # only objects in tile core areas have a segmentation_map_id/are in the MER final catalog
    df = df[df['this_tile_index_is_best'] == True]
    print(len(df))

    # tmp = df[df['object_id'] == -748122396372829839]
    # tmp.to_csv('/home/walml/repos/gz-euclid-datalab/overlap_test.csv', index=False)
    # exit()

    # I specifically picked these tiles for the first upload
    launch_tiles = [
        102015620, 102021061, 102016036, 102021034, 102015615, 102034406,
       102012400, 102013966, 102026083, 102011655, 102027664, 102033849,
       102020090, 102023521, 102018234, 102019150, 102027661, 102016463,
       102022002, 102030421, 102021511, 102031525, 102026603, 102030405,
       102022988, 102016054, 102018712, 102022015, 102022017, 102021057,
       102022027, 102032104, 102028219, 102028213, 102034444, 102032115,
       102022990, 102031550, 102032117, 102022013, 102036817, 102018254,
       102025018, 102023993, 102027667, 102028753, 102029879, 102030997,
       102026063, 102035627
    ]
    already_uploaded_tile_indices_from_notes = launch_tiles.copy()  # will add more here
    # and here is the record of what was actually uploaded
    previous_uploads = pd.concat([pd.read_csv(loc) for loc in glob.glob('/home/walml/repos/gz-euclid-datalab/data/pipeline/zooniverse_upload/*.csv')])
    already_uploaded_tile_indices_from_exports = list(previous_uploads['tile_index'].unique())
    # check it matches what should have been uploaded
    assert set(already_uploaded_tile_indices_from_exports) == set(already_uploaded_tile_indices_from_exports)
    tiles_to_avoid = set(already_uploaded_tile_indices_from_exports) # could pick either
    
    # don't upload those
    df = df[~df['tile_index'].isin(tiles_to_avoid)].reset_index(drop=True)

    # tileset b will be the tiles for the second block of uploads after launch (the first being launch tiles)
    tileset_b = df['tile_index'].unique().tolist()
    # now frozen to avoid accidental changes later
    # json.dump(tileset_b, open('/home/walml/repos/euclid-morphology/upload/tileset_b.json', 'w'))
    # record it here for later adding to 'tiles to avoid'

    print('Galaxies to resize: ', len(df))

    # print(df['object_id'].value_counts())
    assert df['object_id'].value_counts().max() == 1, 'There are duplicate object ids'
    # exit()

    # resize to 424 and save elsewhere
    save_dir = '/home/walml/repos/euclid-morphology/upload/resized'
    im_cols = ['jpg_loc_composite', 'jpg_loc_vis_only', 'jpg_loc_vis_lsb']

    for col in im_cols:

        # adjust for local upload
        df[col] = df[col].apply(lambda x: x.replace(f'/media/home/team_workspaces/Galaxy-Zoo-Euclid/data/pipeline/{version_name}', local_data_dir))
        print(df[col][0])
        # exit()

        # keep tile subdir
        df[col + '_ready_to_resize'] = df[col].apply(lambda x: os.path.exists(x))
        df[col + '_resized'] = df.apply(lambda x: get_resized_loc(x, col), axis=1)
    
    all_images_ready = df[[col + '_ready_to_resize' for col in im_cols]].all(axis=1)
    print('Images ready:', all_images_ready.sum(), 'of ', len(df))
    df = df[all_images_ready].reset_index(drop=True)

    # print(df)
    # exit()

    

    for _, subject in tqdm.tqdm(df.iterrows(), total=len(df), unit='resized'):
        for im_col in im_cols:
            current_loc = subject[im_col]
            new_loc = subject[im_col + '_resized']  # create new cols for the new locations
            # print(current_loc)
            assert current_loc != new_loc, 'You are about to overwrite the original image'
            resize_image(current_loc, new_loc, (424, 424), overwrite=False)
            # print(save_loc)

    # get hashed filename col
    df['id_str'] = get_id_str(df)
    df['!filename'] = df['id_str'].apply(lambda x: get_hash(x, extra_key='_racoon'))

    # make manifest
    df['locations'] = df[[col + '_resized' for col in im_cols]].apply(lambda x: list(x), axis=1)
    df['#campaign'] = 'euclid_challenge'
    df['#upload_date'] = time_utils.current_date()
    df['metadata'] = df[['!filename', '#campaign', '#upload_date']].to_dict(orient='records')   # column of dicts, confusingly


    # subject_set_name = '2024_07_31_euclid_challenge_first_50_tiles_0_19900'
    # df = df[:19900]
    # subject_set_name = '2024_07_31_euclid_challenge_first_50_tiles_19900_end'
    # df = df[19900:]
    # print(df['id_str'])
    # print(df['!filename'])
    # exit()

    # now the upload itself

    # df has 130k entries, and I would prefer to upload tile-by-tile
    # so let's pick 1 tile for each 1k galaxies

    tileset_b = json.load(open('/home/walml/repos/euclid-morphology/upload/tileset_b.json'))
    # shuffle
    np.random.seed(42)
    np.random.shuffle(tileset_b)

    tile_low = 0
    tile_high = 6
    tiles_to_upload = tileset_b[tile_low:tile_high]

    # for galaxies in those tiles, completely shuffle df, will now be uploading random galaxies in random order (from tileset b and those selected tiles only)
    df_to_upload = df[df['tile_index'].isin(tiles_to_upload)]
    df_to_upload = df_to_upload.sample(frac=1, random_state=42).reset_index(drop=True)
    print(df_to_upload['tile_index'].value_counts())


    # for testing
    # subject_set_name = f'{datetime.datetime.now().strftime("%Y_%m_%d")}_euclid_challenge_tileset_b_dev'

    subject_set_name = f'{datetime.datetime.now().strftime("%Y_%m_%d")}_euclid_challenge_tileset_b_tiles_{tile_low}_{tile_high}'
    df.to_csv(f'/home/walml/repos/gz-euclid-datalab/data/pipeline/zooniverse_upload/master_catalog_during_{subject_set_name}.csv', index=False)

    manifest = df_to_upload[['locations', 'metadata']].to_dict(orient='records')
    print(len(manifest))
    print(manifest[0])

    # fast async way
    upload_utils.bulk_upload_subjects(subject_set_name, manifest, project_id='5733')




    # slower way

    # subject_utils.authenticate()

    # project = Project.find(5733)

    # for _, subject in tqdm.tqdm(df.iterrows(), total=len(df), unit='subjects'):

    #     locations = list(subject[[col + '_resized' for col in im_cols]])
    #     metadata = subject[['!filename']]

    #     subject_utils.upload_subject(locations, project, subject_set_name, metadata)

    #     https://github.com/zooniverse/kade/pull/170/files
    #     df['!filename']

