
import logging
import os
import warnings
import hashlib

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits.verify import VerifyWarning
from astropy.nddata import Cutout2D

from bulk_euclid.utils import morphology_utils_ou_mer as m_utils, cutout_utils

import joblib

logging.warning("""
                Setting up query cache at ./joblib. 
                Previous SQL queries for the list of all tiles, and for the list of sources within a given tile, will be re-used for speed. 
                Delete this folder to refresh the cache and make new queries.
                This is CRUCIAL if you switch Euclid environments e.g. from IDR (Q1) to OTF or REG.
                """
)
mem = joblib.Memory('.', verbose=False)

@mem.cache
def get_tiles_in_survey(tile_index=None, bands=None, release_name=None, ra_limits=None, dec_limits=None) -> pd.DataFrame:

    # TODO move release name into survey property, once happy with what it means, if it is per survey?
    query_str = f"""
        SELECT * FROM sedm.mosaic_product 
        WHERE (product_type='DpdMerBksMosaic')
        """
    
    if tile_index is not None:
        query_str += f"AND (tile_index={tile_index})"
    
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

    if 'Euclid' not in locals() or 'Euclid' not in globals():
        logging.critical('"Euclid" class not found, run pipeline_utils.login(cfg) first')
    
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


@mem.cache
def find_relevant_sources_in_tile(cfg, tile_index: int) -> pd.DataFrame:
    # apply our final selection criteria

    """
    segmentation map id query is like:
    SELECT TOP 10 segmentation_map_id
    FROM catalogue.mer_catalogue
    WHERE CAST(segmentation_map_id as varchar) LIKE '102020107%'
    """

    if cfg.sas_environment == 'IDR':
        vis_flux_col = 'flux_vis_1fwhm_aper'  # now renamed with 1FWHM etc
        ext_cols = ''  # not yet available
    else:
        vis_flux_col = 'flux_vis_aper'
        ext_cols = ', flux_g_ext_decam_aper, flux_i_ext_decam_aper, flux_r_ext_decam_aper'
    query_str = f"""
    SELECT object_id, right_ascension, declination, gaia_id, segmentation_area, flux_segmentation, flux_detection_total, {vis_flux_col}, mumax_minus_mag, mu_max, ellipticity, kron_radius, segmentation_map_id {ext_cols}
    FROM catalogue.mer_catalogue
    """

    # non-negative vis flux
    # no cross-match to gaia stars
    # detected in vis
    # not "spurious" (very similar to detected in vis)
    standard_quality_cuts = f"""WHERE {vis_flux_col} > 0
    AND gaia_id IS NULL
    AND vis_det=1
    AND spurious_prob < 0.2
    """
    query_str += standard_quality_cuts

    if cfg.selection_cuts == 'galaxy_zoo':
        logging.info('Applying pre-Q1 volunteer Galaxy Zoo cuts')
        # at least 1200px in area OR ( vis mag < 20.5 (expressed as flux) and at least 200px in area)
        query_str += """AND (segmentation_area > 1200 OR (segmentation_area > 200 AND flux_segmentation > 22.90867652))"""
        # UPDATE - for Q1, changed to 800px. Will see how Zoobot performs on these smaller galaxies.
    elif cfg.selection_cuts == 'galaxy_zoo_generous':
        logging.info('Applying Q1 generous Galaxy Zoo cuts')
        # UPDATE - for Q1, changed to 700px and NO flux cut
        # a hard flux cut of 22.5 (matching strong lensing)? Will see how Zoobot performs on these smaller galaxies.
        # AND (23.9 - 2.5 * LOG10(flux_segmentation)) < 22.5
        # still keep the few bright but small galaxies, for mass completeness
        query_str += """AND (segmentation_area > 700 OR (segmentation_area > 200 AND flux_segmentation > 22.90867652))"""
    elif cfg.selection_cuts == 'space_warps':
        # https://euclidconsortium.slack.com/archives/C05JVCV6TA5/p1728644532577239
        logging.info('Applying lens candidate cuts')
        query_str += """AND segmentation_area > 200
        AND flux_detection_total >= 3.63078
        AND mumax_minus_mag >= -2.6
        AND mu_max >= 15.0
        """
    else:
        raise ValueError(f'Unknown selection cuts {cfg.selection_cuts}')

    # within the tile via segmentation map id
    closing_str = f"""AND CAST(segmentation_map_id as varchar) LIKE '{tile_index}%'
    ORDER BY object_id ASC
    """
    query_str += closing_str
    logging.debug(query_str)

    if 'Euclid' not in locals() or 'Euclid' not in globals():
        logging.critical('"Euclid" class not foun, run pipeline_utils.login(cfg) first')

    # added min segmentation area to remove tiny bright artifacts
    # TODO copy to mer cuts/pipeline
    retries = 0
    df = None
    while retries < cfg.max_retries:
        try:
            if cfg.run_async:
                job = Euclid.launch_job_async(query_str, background=False)
                df = job.get_results().to_pandas()
            else:
                job = Euclid.launch_job(query_str)
                df = job.get_results().to_pandas()
                assert len(df) < 2000, 'Hit query limit, set run_async=True'
            break
        except AttributeError as e:
            logging.info(e)
            logging.info(f'Retrying, {retries}')
        retries += 1
    if df is None:
        raise ValueError(f'Query failed after retries: {query_str}')
    

    logging.info(f"Found {len(df)} query results")
    
    df['tile_index'] = tile_index
    df['mag_segmentation'] = -2.5 * np.log10(df['flux_segmentation']) + 23.9  # for convenience

    return df.reset_index(drop=True)


def download_mosaics(tile_index: int, tiles: pd.DataFrame, download_dir: str) -> pd.DataFrame:
    # save all matching tiles, assuming the tiles catalog only includes relevant data already

    matching_tiles = tiles.query(f'tile_index == {tile_index}')
    assert len(matching_tiles) > 0, f'No matching tiles found for tile index {tile_index}'
    
    matching_tiles = save_euclid_products(matching_tiles, download_dir)  # adds file_loc to downloaded path
    return matching_tiles


def save_euclid_products(df: pd.DataFrame, download_dir: str) -> pd.DataFrame:
    # adds file_loc to downloaded path
    df['file_loc'] = df['file_name'].apply(lambda x: save_euclid_product(x, download_dir))
    return df


def save_euclid_product(product_filename, download_dir) -> str:
    output_loc = os.path.join(download_dir, product_filename)
    if not os.path.isfile(output_loc):
        downloaded_path = Euclid.get_product(file_name=product_filename, output_file=output_loc)[0]  # 0 as one product
        logging.info(f'{product_filename} saved at {downloaded_path}')
    return output_loc


@mem.cache
def get_auxillary_tiles(mosaic_product_oid, auxillary_products: list):

    assert isinstance(auxillary_products, list), 'auxillary_products must be a list'

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
    df['creation_date'] = df['file_name'].apply(lambda x: x.split('_')[-2])  # str, lead by the datetime
    df['tile_index'] = df['file_name'].apply(lambda x: x.split('TILE')[1].split('-')[0])  
    df = df.sort_values(by='creation_date', ascending=False)  # per tile, newest first
    df = df.drop_duplicates(subset=['tile_index', 'product_type_sas'], keep='first').reset_index(drop=True)
    # logging.info(df.iloc[0])
    return df


def get_cutout_loc(base_dir, galaxy, output_format='jpg', version_suffix=None, oneway_hash=False):
    tile_index = str(int(galaxy['tile_index']))
    object_id = str(int(galaxy['object_id'])).replace('-', 'NEG')

    filename_without_format = tile_index + '_' + object_id
    subdir = tile_index
    # e.g. 102159774/102159774_123456.jpg
    if version_suffix is not None:
        subdir = version_suffix + '/' + tile_index
        filename_without_format = filename_without_format + '_' + version_suffix
        # e.g. vis_only/102159774/102159774_123456_vis_only.jpg
        
    if oneway_hash:
        hasher = hashlib.sha256()
        hasher.update(filename_without_format.encode())
        filename_without_format = hasher.hexdigest()

    return os.path.join(base_dir, subdir, filename_without_format + '.' + output_format)


def save_cutouts(cfg, tile_galaxies: pd.DataFrame):
    # assumes the tile has been downloaded and catalogued
    # assumes tile_galaxies includes all/only the bands to load and potentially include
    # print(tile_galaxies.columns.values)
    
    logging.info('loading bands for tile')
    tile_data = {}
    for band in cfg.bands:
        tile_data[band] = fits.getdata(tile_galaxies[f'{band.lower()}_loc'].iloc[0], header=False, memmap=False, decompress_in_memory=True)
    header = fits.getheader(tile_galaxies[f'{cfg.bands[0].lower()}_loc'].iloc[0])
    # vis_loc = tile_galaxies['vis_tile'].iloc[0]
    # nisp_loc = tile_galaxies['y_tile'].iloc[0]
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

        cutout_by_band = {}
        for band in cfg.bands:

            if cfg.field_of_view == 'galaxy_zoo':
                # TODO this bit should use Cutout2D instead
                galaxy.index = galaxy.index.str.upper()  # for the radius estimate
                cutout_by_band[band] = m_utils.extract_cutout_from_array(tile_data[band], galaxy, buff=0, allow_radius_estimate=True)
                galaxy.index = galaxy.index.str.lower()
            else:
                if cfg.field_of_view == 'space_warps':
                    cfg.field_of_view = 20  # arcsec
                assert isinstance(cfg.field_of_view, float) or isinstance(cfg.field_of_view, int)
                # TODO once Cutout2D throughout, I can preserve the header, for now, do .data instead
                cutout_by_band[band] = Cutout2D(tile_data[band], (x_center, y_center), cfg.field_of_view * u.arcsec, wcs=tile_wcs).data 

        
        if cfg.jpg_outputs:  # anything in this list

            # assume jpg_loc_generic key added earlier in catalog creation step
            generic_loc = galaxy['jpg_loc_generic']
            # e.g. jpg_loc/generic/102159774/102159774_123456_generic.jpg

            try:
                # if i == 0:
                #     cutout_subdir = os.path.dirname(generic_loc)
                #     if not os.path.isdir(cutout_subdir):
                #         os.makedirs(cutout_subdir)
                
                # we expect to find the outputs here, see cutout_utils.py
                # skip if all exist and not overwriting. If any missing, don't skip.
                cutout_locs = [generic_loc.replace('generic', output_name) for output_name in cfg.jpg_outputs]
                # e.g. jpg_loc/vis_only/102159774/102159774_123456_vis_only.jpg
                if cfg.overwrite_jpg or (not np.all([os.path.isfile(loc) for loc in cutout_locs])):
                    create_jpgs_within_pipeline(cfg, galaxy, cutout_by_band)

            except AssertionError as e:
                logging.debug(f'skipping galaxy {galaxy["object_id"]} in tile {galaxy["tile_index"]} due to \n{e}')

            
        if cfg.fits_outputs:

            # skip if all exist and not overwriting. If any missing, don't skip.
            if cfg.overwrite_fits or (not os.path.isfile(galaxy['fits_loc'])):

                # lazy copy
                # assume they all are in the same subdir
                if i == 0:
                    cutout_subdir = os.path.dirname(galaxy['fits_loc'])
                    if not os.path.isdir(cutout_subdir):
                        os.makedirs(cutout_subdir)

                # TODO this is a bit of a mess, but I can't use Cutout2D with a header yet
                create_simple_fits(cfg, galaxy, cutout_by_band)



def create_jpgs_within_pipeline(cfg, galaxy, cutout_by_band):

    vis_im = cutout_by_band['VIS']
    y_im = cutout_by_band.get('NIR_Y', None)
    j_im = cutout_by_band.get('NIR_J', None)
    # assume jpg_loc_generic key added earlier in catalog creation step
    save_loc = galaxy['jpg_loc_generic']

    cutout_utils.save_jpg_cutouts(cfg, save_loc, vis_im, y_im, j_im)



def create_simple_fits(cfg, galaxy, cutout_by_band):
    hdr = fits.Header()
    hdr['OBJID'] = galaxy['object_id']
    hdr['TILEIDX'] = galaxy['tile_index']
    hdr['RELEASE'] = galaxy['release_name']
    # hdr.update(cutout_wcs.to_header())  # adds WCS for cutout (vs whole tile)
    header_hdu = fits.PrimaryHDU(header=hdr)
    hdu_list = [header_hdu]

    for band in cfg.bands:
        hdu_list.append(fits.ImageHDU(data=cutout_by_band[band], name=f"{band}_FLUX", header=hdr))
                
    with warnings.catch_warnings():
        # it rewrites my columns to fit the FITS standard by adding HEIRARCH
        warnings.simplefilter('ignore', VerifyWarning)
        fits.HDUList(hdu_list).writeto(galaxy['fits_loc'], overwrite=True)


def login(cfg):
    if os.path.isdir('/media/home/team_workspaces'):
        # two line file, username and password
        # do not commit or put in any team workspace, obviously...
        from astroquery.esa.euclid.core import EuclidClass
        Euclid = EuclidClass(environment=cfg.sas_environment)
        if 'credentials_file' in cfg.keys() and os.path.isfile(cfg.credentials_file):
            Euclid.login(credentials_file=cfg.credentials_file)
        else:
            logging.info('No credentials file found, logging in with username and password')
            Euclid.login()
        globals()['Euclid'] = Euclid  # hack this into everything else, janky but it works and is cleaner than passing it around
    else:
        raise ValueError('Not on DataLabs')
