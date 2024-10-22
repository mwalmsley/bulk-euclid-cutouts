from collections import namedtuple
import logging
import os
import shutil
import warnings
import hashlib

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io.fits.verify import VerifyWarning
from PIL import Image

if os.path.isdir('/media/home/team_workspaces'):  # on datalabs
    from astroquery.esa.euclid.core import Euclid

from bulk_euclid.utils import morphology_utils_ou_mer as m_utils, cutout_utils


Survey = namedtuple('Survey', ['name', 'min_tile_index', 'max_tile_index', 'tile_width', 'tile_overlap'])

# https://www.cosmos.esa.int/documents/10647/12245842/EUCL-EST-ME-8-007_v11_EST_Q1_memo_2022-10-06.pdf
# https://www.cosmos.esa.int/documents/10647/12245842/EUCL-EST-ME-8-014_v10_Q1_product_definition_2023-08-14.pdf
# https://www.cosmos.esa.int/documents/10647/12245842/EUCL-EST-ME-8-018_v1_Q1_fields_definition_2024-07-04.pdf/841e9693-3b5c-89b6-828f-f0faa4f7545b?t=1720533267223

# https://euclid.roe.ac.uk/projects/mer_pf/wiki/Tiling
# tile ids figure

# deep fields are 17x17 arcmin always
# wide tiles are 32x32 arcmin except when edited for special objects

# wide tiles include 2' of overlap (so 30' without overlap, and 1' on each side)
# deep tiles are exactly the same (despite being smaller)

# nominal v1.2 tiling for these surveys, but no data yet except PV-005 below
# EDF_SOUTH = pipeline_utils.Survey(
#     name='edf_south',
#     min_tile_index=0,
#     max_tile_index=int(1.013e8),
#     tile_width=17,
#     tile_overlap=2
# )
# EDF_NORTH = pipeline_utils.Survey(
#     name='edf_north',
#     min_tile_index=int(1.017e8),
#     max_tile_index=int(1.019e8),
#     tile_width=17,
#     tile_overlap=2
# )

# tiles from both south and north in PV-005, likely to be removed in near future (and no catalogues)
EDF_PV = Survey(
    name='edf_pv',
    min_tile_index=0,
    max_tile_index=int(1.013e8),
    tile_width=17,
    tile_overlap=2
)

# no data yet
EDF_FORNAX = Survey(
    name='edf_fornax',
    min_tile_index=int(1.0132e8),
    max_tile_index=int(1.014e8),
    tile_width=17,
    tile_overlap=2
)
    
# 1700 tiles :)
WIDE = Survey(
    name='wide',
    min_tile_index=int(1.02e8),
    max_tile_index=int(1.022e8),
    tile_width=32,
    tile_overlap=2
)
    




def get_tiles_in_survey(tile_index=None, bands=None, release_name=None, ra_limits=None, dec_limits=None) -> pd.DataFrame:

    # TODO move release name into survey property, once happy with what it means, if it is per survey?
    query_str = f"""
        SELECT * FROM sedm.mosaic_product 
        WHERE (product_type='DpdMerBksMosaic')
        """
    
    # if survey is not None:
    #     assert tile_index is None, 'Cannot specify both survey and tile index'
    #     query_str += f"""
    #     AND (tile_index > {survey.min_tile_index}) AND (tile_index < {survey.max_tile_index})
    #     """
    
    if tile_index is not None:
        query_str += f"AND (tile_index={tile_index})"
        # assert survey is None, 'Cannot specify both survey and tile index'
    
    if bands is not None:
        if isinstance(bands, str):
            query_str += f"AND (filter_name='{bands}')"
        else:  # assume listlike
            if len(bands) == 1:
                band = bands[0]
                assert isinstance(band, str), 'Found single band passed as listlike, single band must be a string'
                query_str += f"AND (filter_name='{band}')"
            else:
                query_str += f"AND (filter_name IN {tuple(bands)})"
                
    if ra_limits:
        query_str += f" AND (ra > {ra_limits[0]}) AND (ra < {ra_limits[1]})"
        
    if dec_limits:
        query_str += f" AND (dec > {dec_limits[0]}) AND (dec < {dec_limits[1]})"
        
    if release_name:
        query_str += f" AND release_name='{release_name}'"

    query_str += " ORDER BY tile_index ASC"

    logging.debug(query_str)
    
    # async to avoid 2k max, just note it saves results somewhere on server
    job = Euclid.launch_job_async(query_str, verbose=False, background=False) 
    assert job is not None, 'Query failed with: \n' + query_str
    df = job.get_results().to_pandas()
    
    assert len(df) > 0, 'No results for query with: \n' + query_str
    logging.info(f"Found {len(df)} query results")
    return df


# not used for GZ Euclid
def get_tile_extents_fov(tiles: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cols ['ra_min', 'ra_max', 'dec_min', 'dec_max'] by unpacking the "fov" tile metadata column
    fov = Field of View, the corners of the tile in RA and Dec
    Thanks to Kristin Remmelgas

    Args:
        tiles (pd.DataFrame): table of MER mosaic products with an 'fov' column in ADQL format

    Returns:
        pd.DataFrame: same as input, but with ['ra_min', 'ra_max', 'dec_min', 'dec_max'] columns showing edges of tile FoV
    """
    
    tiles = tiles.copy()
    float_fovs = tiles['fov'].apply(lambda x: np.array(x[1:-1].split(", ")).astype(np.float64)) # from one big string to arrays of floats
    array_fovs = np.array(float_fovs.values.tolist()) #from pandas series to numpy array
    ras = array_fovs[:, ::2]
    decs = array_fovs[:, 1::2]

    tiles['ra_min'] = np.min(ras, axis=1)
    tiles['ra_max'] = np.max(ras, axis=1)
    tiles['dec_min'] = np.min(decs, axis=1)
    tiles['dec_max'] = np.max(decs, axis=1)
    return tiles


def find_relevant_sources_in_tile(cfg, tile):
    # apply our final selection criteria

    """
    segmentation map id query is like:
    SELECT TOP 10 segmentation_map_id
    FROM catalogue.mer_catalogue
    WHERE CAST(segmentation_map_id as varchar) LIKE '102020107%'
    """

    query_str = f"""
    SELECT object_id, right_ascension, declination, gaia_id, segmentation_area, flux_segmentation, flux_vis_aper, ellipticity, kron_radius, segmentation_map_id, flux_g_ext_decam_aper, flux_i_ext_decam_aper, flux_r_ext_decam_aper
    FROM catalogue.mer_catalogue
    """

    # non-negative vis flux
    # no cross-match to gaia stars
    # detected in vis
    # not "spurious" (very similar to detected in vis)
    standard_quality_cuts = """WHERE flux_vis_aper > 0
    AND gaia_id IS NULL
    AND vis_det=1
    AND spurious_prob < 0.2
    """
    query_str += standard_quality_cuts

    if cfg.selection_cuts == 'galaxy_zoo':
        logging.info('Applying Galaxy Zoo cuts')
        # at least 1200px in area OR ( vis mag < 20.5 (expressed as flux) and at least 200px in area)
        query_str += """AND (segmentation_area > 1200 OR (segmentation_area > 200 AND flux_segmentation > 22.90867652))
        """
    elif cfg.selection_cuts == 'lens_candidates':
        logging.info('Applying lens candidate cuts')
        query_str += """AND segmentation_area > 100
        AND flux_r_ext_decam_aper > 3.630780547701008
        AND flux_r_ext_decam_aper < 229.08676527677702
        AND flux_g_ext_decam_aper < 36.307805477010085
        AND flux_i_ext_decam_aper < 190.54607179632464
        AND flux_i_ext_decam_aper > 1.4454397707459257
        AND (flux_g_ext_decam_aper / flux_i_ext_decam_aper) > 0.01
        AND (flux_g_ext_decam_aper / flux_i_ext_decam_aper) < 0.19054607179632474
        AND (flux_g_ext_decam_aper / flux_r_ext_decam_aper) > 0.06309573444801933
        AND (flux_g_ext_decam_aper / flux_r_ext_decam_aper) < 0.5754399373371569
        """
        # AND (flux_g_ext_decam_aper - flux_i_ext_decam_aper) < 5
        # AND (flux_g_ext_decam_aper - flux_i_ext_decam_aper) > 1.8
        # AND (flux_g_ext_decam_aper - flux_r_ext_decam_aper) < 3
        # AND (flux_g_ext_decam_aper - flux_r_ext_decam_aper) > 0.6

    # within the tile via segmentation map id
    closing_str = f"""AND CAST(segmentation_map_id as varchar) LIKE '{tile['tile_index']}%'
    ORDER BY object_id ASC
    """
    query_str += closing_str
    logging.debug(query_str)

    # added min segmentation area to remove tiny bright artifacts
    # TODO copy to mer cuts/pipeline
    retries = 0
    while retries < cfg.max_retries:
        try:
            if cfg.run_async:
                output_tmpfile = f'tmpfile_{np.random.rand(int(1e8))}.csv'
                job = Euclid.launch_job_async(query_str, background=False, output_file=output_tmpfile, output_format='csv')
                df = pd.read_csv(output_tmpfile)
                shutil.rm(output_tmpfile)
            else:
                job = Euclid.launch_job(query_str)
                df = job.get_results().to_pandas()
                assert len(df) < 2000, 'Hit query limit, set run_async=True'
            break
        except AttributeError as e:
            logging.info(e)
            logging.info(f'Retrying, {retries}')
        retries += 1

    logging.info(f"Found {len(df)} query results")
    
    df['tile_index'] = tile['tile_index']
    df['mag_segmentation'] = -2.5 * np.log10(df['flux_segmentation']) + 23.9  # for convenience

    return df.reset_index(drop=True)


def download_mosaics(tile_index: int, tiles: pd.DataFrame, download_dir: str):
    # save all matching tiles, assuming the tiles catalog only includes relevant data already

    matching_tiles = tiles.query(f'tile_index == {tile_index}')
    assert len(matching_tiles) > 0, f'No matching tiles found for tile index {tile_index}'
    
    matching_tiles = save_euclid_products(matching_tiles, download_dir)  # adds file_loc to downloaded path
    return matching_tiles


def save_euclid_products(df: pd.DataFrame, download_dir: str):
    # adds file_loc to downloaded path
    df['file_loc'] = df['file_name'].apply(lambda x: save_euclid_product(x, download_dir))
    return df


def save_euclid_product(product_filename, download_dir):
    output_loc = os.path.join(download_dir, product_filename)
    if not os.path.isfile(output_loc):
        downloaded_path = Euclid.get_product(file_name=product_filename, output_file=output_loc)[0]  # 0 as one product
        logging.info(f'{product_filename} saved at {downloaded_path}')
    return output_loc


def get_auxillary_tiles(mosaic_product_oid, auxillary_products=['MERPSF', 'MERRMS', 'MERBKG']):

    for aux in auxillary_products:
        assert aux in ['MERPSF', 'MERRMS', 'MERBKG', 'MERFLG'], f'Unknown or unsupported auxillary product {aux}'

    query_str = f"""
    SELECT * FROM sedm.aux_mosaic 
    WHERE (mosaic_product_oid={mosaic_product_oid})
    """
    if len(auxillary_products) > 1:
        query_str += f"AND (product_type_sas IN {tuple(auxillary_products)})"
    elif len(auxillary_products) == 1:
        query_str += f"AND (product_type_sas='{auxillary_products[0]}')"

    df = Euclid.launch_job(query_str).get_results().to_pandas()


    """
    Can sometimes have multiple auxillary tiles with the same mosaic_product_oid
    EUC_MER_BGSUB-MOSAIC-VIS_TILE102159774-3EAE6B_20240707T183311.123620Z_00.00.fits
    EUC_MER_BGSUB-MOSAIC-VIS_TILE102159774-FE2962_20240806T043542.352405Z_00.00.fits
    For now, take the most recent one
    """
    df['creation_date'] = df['file_name'].apply(lambda x: x.split('_')[-2])
    df['tile_index'] = df['file_name'].apply(lambda x: int(x.split('TILE')[1].split('-')[-1]))
    df = df.sort_values(by='creation_date', ascending=False)  # per tile, newest first
    df = df.drop_duplicates(subset='tile_index', keep='first').reset_index(drop=True)
    logging.info(df.iloc[0])
    return df


def get_cutout_loc(base_dir, galaxy, output_format='jpg', version_suffix=None, oneway_hash=False):
    tile_index = str(int(galaxy['tile_index']))
    object_id = str(int(galaxy['object_id'])).replace('-', 'NEG')
    subdir = tile_index
    filename_without_format = tile_index + '_' + object_id
    if version_suffix is not None:
        filename_without_format = filename_without_format + '_' + version_suffix
        
    if oneway_hash:
        hasher = hashlib.sha256()
        hasher.update(filename_without_format.encode())
        filename_without_format = hasher.hexdigest()
    return os.path.join(base_dir, subdir, filename_without_format + '.' + output_format)


def save_cutouts(vis_loc, nisp_loc, tile_galaxies: pd.DataFrame, output_format:str='jpg', overwrite:bool=False, allow_radius_estimate:bool=True):
    
    logging.info('loading tile')
    vis_data, header = fits.getdata(vis_loc, header=True, memmap=False, decompress_in_memory=True)
    nisp_data = fits.getdata(nisp_loc, header=False, memmap=False, decompress_in_memory=True)
    logging.info('tile loaded')
    
    tile_galaxies = tile_galaxies.reset_index(drop=True)
    
    tile_wcs = WCS(header)

    for i, galaxy in tile_galaxies.iterrows():
        
        if i % 100 == 0:
            logging.info(f'galaxy {i} of {len(tile_galaxies)}')
                  
        c = SkyCoord(galaxy['right_ascension'], galaxy['declination'], frame='icrs', unit="deg")
        x_center, y_center = tile_wcs.world_to_pixel(c)

         # these are the pixel coordinates of the galaxy wrt. the tile.
         # for big sources, it might be possible to be centered off the edge of the tile?
        galaxy['x_center'] = x_center  
        galaxy['y_center'] = y_center
        galaxy['log_segmentation_area'] = np.log10(galaxy['segmentation_area'])
        galaxy['log_kron_radius'] = np.log10(galaxy['kron_radius'])
        
        if output_format == 'jpg':
            
            try:
                
                # really should make this a class...
                # TODO might be different for lens search
                cutout_locs = galaxy[['jpg_loc_composite', 'jpg_loc_vis_only', 'jpg_loc_vis_lsb']]

                # assume they all are in the same subdir
                if i == 0:
                    cutout_subdir = os.path.dirname(cutout_locs['jpg_loc_composite'])
                    if not os.path.isdir(cutout_subdir):
                        os.makedirs(cutout_subdir)
    
                
                if np.all([os.path.isfile(loc) for loc in cutout_locs]) and not overwrite:
                    continue

                galaxy.index = galaxy.index.str.upper()
                vis_cutout = m_utils.extract_cutout_from_array(vis_data, galaxy, buff=0, allow_radius_estimate=allow_radius_estimate)
                nisp_cutout = m_utils.extract_cutout_from_array(nisp_data, galaxy, buff=0, allow_radius_estimate=allow_radius_estimate)
                galaxy.index = galaxy.index.str.lower()
                # extremely lazy coding, I should make a class or smth
                # print(f'got slice {vis_cutout.shape}, {nisp_cutout.shape}')

                cutout = cutout_utils.make_composite_cutout(vis_cutout, nisp_cutout)
                Image.fromarray(cutout).save(galaxy['jpg_loc_composite'])

                cutout = m_utils.make_vis_only_cutout(vis_cutout)
                Image.fromarray(cutout).save(galaxy['jpg_loc_vis_only'])

                # magic params from tinkering
                cutout = cutout_utils.make_lsb_cutout(vis_cutout, stretch=20, power=0.5)
                # print('vis lsb ready')
                Image.fromarray(cutout).save(galaxy['jpg_loc_vis_lsb'])
                # print('vis lsb saved')
                
            except AssertionError as e:
                galaxy.index = galaxy.index.str.lower()
                print(f'skipping galaxy {galaxy["object_id"]} in tile {galaxy["tile_index"]} due to \n{e}')

            
        elif (output_format == 'fits.gz') or (output_format == 'fits'):

    

            # lazy copy
            # assume they all are in the same subdir
            if i == 0:
                cutout_subdir = os.path.dirname(galaxy['fits_loc'])
                if not os.path.isdir(cutout_subdir):
                    os.makedirs(cutout_subdir)


            # lazy copy

            galaxy.index = galaxy.index.str.upper()
            vis_cutout, cutout_wcs = m_utils.extract_cutout_from_fits(vis_data, tile_wcs, galaxy, buff=0, allow_radius_estimate=allow_radius_estimate)
            nisp_cutout, _ = m_utils.extract_cutout_from_fits(nisp_data, tile_wcs, galaxy, buff=0, allow_radius_estimate=allow_radius_estimate)
            # galaxy.index = galaxy.index.str.lower()

    
            hdr = fits.Header()
            # hdr['COMMENT'] = json.dumps(galaxy[['object_id', 'tile_index', 'release_name']].to_dict())
            hdr.update(galaxy[['OBJECT_ID', 'TILE_INDEX', 'RELEASE_NAME']].to_dict())
            hdr.update(cutout_wcs.to_header())  # adds WCS for cutout (vs whole tile)
            header_hdu = fits.PrimaryHDU(header=hdr)
            
            vis_hdu = fits.ImageHDU(data=vis_cutout, name="VIS_FLUX_MICROJANSKY", header=hdr)
            nisp_hdu = fits.ImageHDU(data=nisp_cutout, name="NISP_Y_FLUX_MICROJANSKY", header=hdr)
            
            hdul = fits.HDUList([header_hdu, vis_hdu, nisp_hdu])
            
            with warnings.catch_warnings():
                # it rewrites my columns to fit the FITS standard by adding HEIRARCH
                warnings.simplefilter('ignore', VerifyWarning)
                hdul.writeto(galaxy['FITS_LOC'], overwrite=True)
        else:
            raise ValueError(f'{output_format} format not recognised')


def login():

    if os.path.isdir('/media/home/team_workspaces'):
        from astroquery.esa.euclid.core import Euclid
        # two line file, username and password
        # do not commit or put in any team workspace, obviously...
        Euclid.login(credentials_file='/media/user/_credentials/euclid_login.txt')
    else:
        raise ValueError('Not on DataLabs')
