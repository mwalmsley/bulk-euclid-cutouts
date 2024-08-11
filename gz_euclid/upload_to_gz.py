import os

from shared_astro_utils import subject_utils, upload_utils, time_utils
import tqdm
import hashlib
from PIL import Image
from panoptes_client import Project
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

    # TODO download cutouts/catalogs again from datalabs and rerun, we need the lsb also



    # df_loc = '/home/walml/repos/euclid-morphology/datalabs/data/pipeline/v2_challenge_launch_local/catalogs/_master_catalog.csv'
    df_loc = '/home/walml/repos/euclid-morphology/datalabs/data/pipeline/v2_challenge_launch/catalogs/_master_catalog.csv'
    local_data_dir = '/home/walml/repos/euclid-morphology/datalabs/data/pipeline/v2_challenge_launch'


    df = pd.read_csv(df_loc)

    # apply the final filter for galaxies good to classify but not to upload
    # for now, anything about 25k pixels
    df = df[df['segmentation_area'] < 25000]

    # select which tiles to upload
    # selected toa void overlaps, see mer notebook
    
    valid_tile_indices = [
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
    df = df[df['tile_index'].isin(valid_tile_indices)].reset_index(drop=True)
    print('Galaxies to resize: ', len(df))

    # print(df['object_id'].value_counts())
    assert df['object_id'].value_counts().max() == 1, 'There are duplicate object ids'
    # exit()


    # resize to 424 and save elsewhere
    save_dir = '/home/walml/repos/euclid-morphology/upload/resized'
    im_cols = ['jpg_loc_composite', 'jpg_loc_vis_only', 'jpg_loc_vis_lsb']

    for col in im_cols:

        # adjust for local upload
        df[col] = df[col].apply(lambda x: x.replace('/media/home/team_workspaces/Galaxy-Zoo-Euclid/data/pipeline/v2_challenge_launch', local_data_dir))
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

    subject_set_name = '2024_07_31_euclid_challenge_first_50_tiles_19900_end'
    df = df[19900:]


    # print(df['id_str'])
    # print(df['!filename'])
    # exit()
    df.to_csv(f'/home/walml/repos/euclid-morphology/upload/master_catalog_during_{subject_set_name}.csv', index=False)
    # manifest = df[['locations', 'metadata']].to_dict(orient='records')
    # print(manifest[0])
    # exit()

        # subject_utils.authenticate()

    # project = Project.find(5733)

 
    # upload_utils.bulk_upload_subjects(subject_set_name, manifest, project_id='5733')

    # for _, subject in tqdm.tqdm(df.iterrows(), total=len(df), unit='subjects'):

    #     locations = list(subject[[col + '_resized' for col in im_cols]])
    #     metadata = subject[['!filename']]

    #     subject_utils.upload_subject(locations, project, subject_set_name, metadata)

    #     https://github.com/zooniverse/kade/pull/170/files
    #     df['!filename']

