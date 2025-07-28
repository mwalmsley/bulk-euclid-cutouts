
import logging
import os
import glob
import warnings
import hashlib
from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf
# from omegaconf.errors import ConfigAttributeError
# from omegaconf.listconfig import ListConfig

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
                Delete this folder to refresh the cache and make new queries.
                This is CRUCIAL if the underlying data changes, e.g. new tiles are added.
                """
)
mem = joblib.Memory('.', verbose=False)


# setting up like nested json db, with this schema

@dataclass
class Mosaic:
    path: str
    _data: Optional[np.ndarray] = None  # hidden attr, loaded on demand and then stored

    # def __post_init__(self):
        # this would be better but it triggers a super slow datalabs first read

    @property # public attr for accessing _data
    def data(self):
        if self.path and self._data is None:
            assert os.path.isfile(self.path), f'Mosaic path {self.path} does not exist'
            self._data = load_observation_fits(self.path)
        return self._data

@dataclass
class Observation:
    band: str # e.g. VIS, NIR_Y, NIR_J, NIR_H
    instrument: str = None  # e.g. NISP, VIS
    BGMOD: Optional[Mosaic] = None
    BGSUB: Optional[Mosaic] = None
    RMS: Optional[Mosaic] = None
    PSF: Optional[Mosaic] = None  # catalog psf

@dataclass
class Tile:
    tile_index: int
    # ra: float
    # dec: float
    release_name: str
    # hate caps but it's convention
    VIS: Optional[Observation] = None
    NIR_Y: Optional[Observation] = None
    NIR_J: Optional[Observation] = None
    NIR_H: Optional[Observation] = None
    mer_final_catalog: Optional[str] = None
    # mer_morphology_catalog: str = None


@mem.cache # again we assume the release directory changes rarely, so caching is fine
def get_path_if_exists(search_str: str) -> str:
    """Check if a path exists, return it if it does, else return None."""
    try:
        return list(glob.glob(search_str))[0]
    except IndexError:
        logging.info('Path not found: {}'.format(search_str))
        return None

def find_available_tiles(cfg: OmegaConf):

    # tiles = pipeline_utils.get_tiles_in_survey(bands=cfg.bands, release_name=cfg.release_name)  # F-003_240321 recently appeared

    if cfg.release_name == 'Q1_R1':
        release_dir = '/media/home/data/euclid_q1/Q1_R1'
    elif cfg.release_name == 'RR2_R1':
        release_dir = '/media/home/data/euclid_reg/REGREPROC2_R1'
    else:
        raise ValueError('Release name not recognised for tile search: {}'.format(cfg.release_name))

    # all subfolders in the release_dir, each name is a tile_index
    tile_indices = [ int(os.path.basename(f.path)) for f in os.scandir(release_dir + '/MER') if f.is_dir() ]
    tile_indices = sorted(tile_indices)
    logging.info(f'Found {len(tile_indices)} tiles e.g. {tile_indices[0]}')

    if cfg.max_tiles and len(tile_indices) > cfg.max_tiles:
        logging.info(f'Randomly subselecting {cfg.max_tiles} tiles')
        tile_indices = np.random.choice(tile_indices, cfg.max_tiles, replace=False).tolist()

    tiles = []
    for tile_index in tile_indices:
        tile = create_tile_object(cfg, release_dir, tile_index)
        tiles.append(tile)

    if len(tiles) == 0:
        logging.error('No tiles found, exiting')
        exit(1)

    logging.info('Tile list created with {} tiles'.format(len(tiles)))
    
    return tiles

@mem.cache  # we assume the release directory changes rarely, so caching is fine
def create_tile_object(cfg, release_dir, tile_index):
    tile = Tile(tile_index=tile_index, release_name=cfg.release_name)
    tile.mer_final_catalog = get_path_if_exists(f'{release_dir}/MER_FINAL_CATALOG/{tile_index}/EUC_MER_FINAL-CAT_TILE{tile_index}*.fits')
        # fill columns for paths/existence to mosaics (all bands), MER final/morphology catalogs, value-added products
    for (instrument, band) in [('VIS', 'VIS'), ('NISP', 'NIR_Y'), ('NISP', 'NIR_J'), ('NISP', 'NIR_H')]:  # could add EXT here
        band_w_hyphen = band.replace('_', '-')  # python can't use hyphens in variable names
        mosaic = Observation(band=band)
        mosaic.instrument = instrument

        mosaic.BGMOD = Mosaic(get_path_if_exists(f'{release_dir}/MER/{tile_index}/{instrument}/EUC_MER_BGMOD-{band_w_hyphen}_TILE{tile_index}-*.fits'))
        mosaic.BGSUB = Mosaic(get_path_if_exists(f'{release_dir}/MER/{tile_index}/{instrument}/EUC_MER_BGSUB-MOSAIC-{band_w_hyphen}_TILE{tile_index}-*.fits'))
        mosaic.RMS = Mosaic(get_path_if_exists(f'{release_dir}/MER/{tile_index}/{instrument}/EUC_MER_MOSAIC-{band_w_hyphen}-RMS_TILE{tile_index}-*.fits'))
        mosaic.PSF = Mosaic(get_path_if_exists(f'{release_dir}/MER/{tile_index}/{instrument}/EUC_MER_CATALOG-PSF-{band_w_hyphen}_TILE{tile_index}-*.fits'))

            # for now, only use if all required data products exist
        if all([getattr(mosaic, key, False) for key in cfg.data_products]):  # e.g. BGSUB, BGMOD, RMS. String is Truthy.
            setattr(tile, band, mosaic)
        else:
            logging.warning(f'Skipping mosaic as not all data products exist, for tile {tile_index}, instrument {instrument}, band {band}: {mosaic}')
    return tile



def find_relevant_sources_in_tile(cfg, df: pd.DataFrame) -> pd.DataFrame:
    # apply our final selection criteria
    # df should be mer catalogue for that tile

    """
    segmentation map id query is like:
    SELECT TOP 10 segmentation_map_id
    FROM catalogue.mer_catalogue
    WHERE CAST(segmentation_map_id as varchar) LIKE '102020107%'
    """

    # if cfg.cfg.release_name in ['Q1_R1', 'RR2_R1']:
    vis_flux_col = 'flux_vis_1fwhm_aper'  # now renamed with 1FWHM etc
        # ext_cols = []  # not yet available
    # elif cfg.sas_environment == 'PDR':
        # vis_flux_col = 'flux_vis_1fwhm_aper'
        # ext_cols = ['flux_g_ext_decam_1fwhm_aper', 'flux_i_ext_decam_1fwhm_aper', 'flux_r_ext_decam_1fwhm_aper']
    # else:
        # vis_flux_col = 'flux_vis_aper'
        # ext_cols = ['flux_g_ext_decam_aper', 'flux_i_ext_decam_aper', 'flux_r_ext_decam_aper']

    # only relevant columns
    required_cols = ['object_id', 'right_ascension', 'declination', 'gaia_id', 'segmentation_area', 'flux_segmentation', 'flux_detection_total', vis_flux_col, 'mumax_minus_mag', 'mu_max', 'ellipticity', 'kron_radius', 'segmentation_map_id', 'vis_det', 'spurious_prob']
    # optional_cols = ext_cols
    # df = df[relevant_cols]
    assert all([col in df.columns for col in required_cols]), f'Missing columns in dataframe: {set(required_cols) - set(df.columns)}'

    # apply as pandas cuts
    df = df.query(vis_flux_col + ' > 0')  # non-negative vis flux
    # df = df.query('gaia_id.isnull()')  # no cross-match to gaia stars - now replaced
    # https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/mermorphologycookbook.html#point-like-probability
    # use point_like_prob to reject stars instead
    df = df[(df['point_like_prob'] < 0.5) | (df['segmentation_area'] > 10000)]  # point-like is nan for very small objects (should reject) and very large (should keep)
    df = df.query('vis_det == 1')  # detected in vis
    df = df.query('spurious_prob < 0.2')  # not "spurious" (very similar to detected in vis)

    if cfg.selection_cuts == 'galaxy_zoo':
        logging.info('Applying pre-Q1 volunteer Galaxy Zoo cuts')
        above_min_area = df['segmentation_area'] > 1200  # at least 1200px in area
        above_min_flux = (df['flux_segmentation'] > 22.90867652) & (df['segmentation_area'] > 200)
        df = df[above_min_area | above_min_flux]  # keep galaxies that are either large enough or bright enough
        # at least 1200px in area OR ( vis mag < 20.5 (expressed as flux) and at least 200px in area)
        # UPDATE - for Q1, changed to 700px. Will see how Zoobot performs on these smaller galaxies.
    elif cfg.selection_cuts == 'galaxy_zoo_generous':
        logging.info('Applying Q1 generous Galaxy Zoo cuts')
        # UPDATE - for Q1, changed to 700px and NO flux cut
        # a hard flux cut of 22.5 (matching strong lensing)? Will see how Zoobot performs on these smaller galaxies.
        # AND (23.9 - 2.5 * LOG10(flux_segmentation)) < 22.5
        # still keep the few bright but small galaxies, for mass completeness
        above_min_area = df['segmentation_area'] > 700  # at least 700px in area
        above_min_flux = (df['flux_segmentation'] > 22.90867652) & (df['segmentation_area'] > 200)
        df = df[above_min_area | above_min_flux]  # keep galaxies that are either large enough or bright enough
    elif cfg.selection_cuts == 'space_warps':
        # https://euclidconsortium.slack.com/archives/C05JVCV6TA5/p1728644532577239
        logging.info('Applying lens candidate cuts')
        df = df.query('segmentation_area > 200')  
        df = df.query('flux_detection_total >= 3.63078')  # flux detection total, not segmentation
        df = df.query('mumax_minus_mag >= -2.6')  # mumax minus mag, not mu_max
        df = df.query('mu_max >= 15.0')  # mu_max, not mumax minus mag
    else:
        raise ValueError(f'Unknown selection cuts {cfg.selection_cuts}')

    # within the tile via segmentation map id
    tile_index = df['segmentation_map_id'].apply(lambda x: int(str(x)[:9]))

    # for convenience
    df['mag_segmentation'] = -2.5 * np.log10(df['flux_segmentation']) + 23.9  # for convenience
    df['tile_index'] = tile_index  # add tile index column

    if df.empty:
        logging.warning('No relevant sources found in tile, returning empty DataFrame')
        return df

    df = df.sort_values(by='object_id')  # sort by object id, for consistency
    df = df.reset_index(drop=True)
    logging.info(f"Found {len(df)} relevant sources")
    logging.info(f'First galaxy: {df.iloc[0]["object_id"]}, tile {df.iloc[0]["tile_index"]}')

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


def load_observation_fits(mosaic_path: str, header: bool = False) -> np.ndarray:
    logging.debug(f'Loading mosaic {os.path.basename(mosaic_path)} from {mosaic_path}')
    mosaic = fits.getdata(mosaic_path, header=header, memmap=False, decompress_in_memory=False)  # type: ignore
    logging.debug(f'Loaded mosaic {os.path.basename(mosaic_path)}, shape: {mosaic.shape}')
    return mosaic
    # https://docs.astropy.org/en/latest/io/fits/api/files.html
    # memmap allows access to small segments without loading the whole file into memory
    # decompress_in_memory probably has no effect on uncompressed .fits? 

def save_cutouts(cfg, tile: Tile, tile_galaxies: pd.DataFrame):

    # assume we always have VIS and use BGSUB tile as our reference for WCS etc
    header = fits.getheader(tile.VIS.BGSUB.path)

    tile_wcs = WCS(header)

    for i, galaxy in tile_galaxies.iterrows():
        
        if i % 1000 == 0 or i == 1 or i == 2:  # useful for checking how long it takes to load mosaics
            logging.info(f'galaxy {i} of {len(tile_galaxies)}, {galaxy["object_id"]} in tile {tile.tile_index}')
                  
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

            # create the flux array

            # background-subtracted flux
            # e.g. tile.NIR_Y.BGSUB.data
            flux = tile.__dict__[band].BGSUB.data
            if cfg.add_bkg:  # twice as long to make cutouts and very very small effect for sources smaller than a few tens of arcsec
                subtracted_bkg = tile.__dict__[band].BGMOD.data
                flux = flux + subtracted_bkg

            # set field of view for the slice from the flux array

            logging.debug('Getting FoV')
            if cfg.field_of_view == 'galaxy_zoo':  # use segmentation map sizing
                source_r_max = m_utils.estimate_source_r_max(galaxy)
                # source_r_max is half cutout width in pixels
                # so source_r_max * 2 / 10 is cutout width in arcsec
                field_of_view = source_r_max * 0.2 * u.arcsec  
                
            elif cfg.field_of_view == 'space_warps':  # use standard fixed sizing of 20 arcsec
                field_of_view = 20 * u.arcsec
            else:  # assume cfg.field_of_view is a number
                assert isinstance(cfg.field_of_view, float) or isinstance(cfg.field_of_view, int)
                field_of_view = cfg.field_of_view  * u.arcsec
                
            # TODO I could preserve the header, for now, do .data instead
            # use cutout2D to apply the slice
            logging.debug(f'Creating cutout for {band} band with field of view {field_of_view}')
            cutout_by_band[band] = Cutout2D(flux, (x_center, y_center), field_of_view, wcs=tile_wcs).data

        
        if cfg.jpg_outputs:  # anything in this list

            # assume jpg_loc_generic key added earlier in catalog creation step
            generic_loc = galaxy['jpg_loc_generic']
            # e.g. jpg_loc/generic/102159774/102159774_123456_generic.jpg

            try:
                
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





# # @mem.cache
# def get_tiles_in_survey(tile_index=None, bands=None, release_name=None, ra_limits=None, dec_limits=None) -> pd.DataFrame:

#     # TODO move release name into survey property, once happy with what it means, if it is per survey?
#     query_str = f"""
#         SELECT * FROM sedm.mosaic_product 
#         WHERE (product_type='DpdMerBksMosaic')
#         """
    
#     if tile_index is not None:
#         query_str += f"AND (tile_index={tile_index})"
    
#     if bands is not None:
#         if isinstance(bands, str):
#             query_str += f"AND (filter_name='{bands}')"
#         else:  # assume listlike
#             if len(bands) == 1:
#                 band = bands[0]
#                 assert isinstance(band, str), 'Found single band passed as listlike, single band must be a string'
#                 query_str += f"AND (filter_name='{band}')"
#             else:
#                 query_str += f"AND (filter_name IN {tuple(bands)})"
                
#     if ra_limits:
#         query_str += f" AND (ra > {ra_limits[0]}) AND (ra < {ra_limits[1]})"
        
#     if dec_limits:
#         query_str += f" AND (dec > {dec_limits[0]}) AND (dec < {dec_limits[1]})"
        
#     if release_name:
#         query_str += f" AND release_name='{release_name}'"

#     query_str += " ORDER BY tile_index ASC"

#     logging.debug(query_str)

#     # this doesn't work as expected and I don't know why
#     # it always fails to find Euclid
#     # if 'Euclid' not in locals() or 'Euclid' not in globals():
#     try:
#         Euclid
#     except NameError:
#         logging.critical('"Euclid" class not found, run pipeline_utils.login(cfg) first')
    

#     # async to avoid 2k max, just note it saves results somewhere on server
#     job = Euclid.launch_job_async(query_str, verbose=False, background=False) 
#     assert job is not None, 'Query failed with: \n' + query_str
#     df = job.get_results().to_pandas()
    
#     assert len(df) > 0, 'No results for query with: \n' + query_str
#     logging.info(f"Found {len(df)} query results")
#     return df


# not used for GZ Euclid
# def get_tile_extents_fov(tiles: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds cols ['ra_min', 'ra_max', 'dec_min', 'dec_max'] by unpacking the "fov" tile metadata column
#     fov = Field of View, the corners of the tile in RA and Dec
#     Thanks to Kristin Remmelgas

#     Args:
#         tiles (pd.DataFrame): table of MER mosaic products with an 'fov' column in ADQL format

#     Returns:
#         pd.DataFrame: same as input, but with ['ra_min', 'ra_max', 'dec_min', 'dec_max'] columns showing edges of tile FoV
#     """
    
#     tiles = tiles.copy()
#     float_fovs = tiles['fov'].apply(lambda x: np.array(x[1:-1].split(", ")).astype(np.float64)) # from one big string to arrays of floats
#     array_fovs = np.array(float_fovs.values.tolist()) #from pandas series to numpy array
#     ras = array_fovs[:, ::2]
#     decs = array_fovs[:, 1::2]

#     tiles['ra_min'] = np.min(ras, axis=1)
#     tiles['ra_max'] = np.max(ras, axis=1)
#     tiles['dec_min'] = np.min(decs, axis=1)
#     tiles['dec_max'] = np.max(decs, axis=1)
#     return tiles



# def login(cfg):
#     if os.path.isdir('/media/home/team_workspaces'):
#         # two line file, username and password
#         # do not commit or put in any team workspace, obviously...
#         from astroquery.esa.euclid.core import EuclidClass
#         Euclid = EuclidClass(environment=cfg.sas_environment)
#         logging.info(cfg)
#         try:
#             logging.info(f'Logging in with credentials file {cfg.credentials_file}')
#             assert os.path.isfile(cfg.credentials_file), f'Credentials file not found at {cfg.credentials_file}'
#             Euclid.login(credentials_file=cfg.credentials_file)
#         except ConfigAttributeError:
#         # if OmegaConf.is_missing(cfg, "credentials_file"):  # this actually only catches "???", not simply no key
#             logging.info('No cfg.credentials_file, logging in with username and password')
#             Euclid.login()
#         globals()['Euclid'] = Euclid  # hack this into everything else, janky but it works and is cleaner than passing it around
#     else:
#         raise ValueError('Not on DataLabs')


# def download_mosaics(tile_index: int, tiles: pd.DataFrame, download_dir: str) -> pd.DataFrame:
#     # save all matching tiles, assuming the tiles catalog only includes relevant data already

#     matching_tiles = tiles.query(f'tile_index == {tile_index}')
#     assert len(matching_tiles) > 0, f'No matching tiles found for tile index {tile_index}'
    
#     matching_tiles = save_euclid_products(matching_tiles, download_dir)  # adds file_loc to downloaded path
#     return matching_tiles


# def save_euclid_products(df: pd.DataFrame, download_dir: str) -> pd.DataFrame:
#     # adds file_loc to downloaded path
#     df['file_loc'] = df['file_name'].apply(lambda x: save_euclid_product(x, download_dir))
#     return df


# def save_euclid_product(product_filename, download_dir) -> str:
#     output_loc = os.path.join(download_dir, product_filename)
#     if not os.path.isfile(output_loc):
#         downloaded_path = Euclid.get_product(file_name=product_filename, output_file=output_loc)[0]  # 0 as one product
#         logging.info(f'{product_filename} saved at {downloaded_path}')
#     return output_loc


# @mem.cache
# def get_auxillary_tiles(mosaic_product_oid, auxillary_products: list):

#     assert isinstance(auxillary_products, list) or isinstance(auxillary_products, ListConfig), 'auxillary_products must be a list, is {} ({})'.format(auxillary_products, type(auxillary_products))

#     for aux in auxillary_products:
#         assert aux in ['MERPSF', 'MERRMS', 'MERBKG', 'MERFLG'], f'Unknown or unsupported auxillary product {aux}'

#     query_str = f"""
#     SELECT * FROM sedm.aux_mosaic 
#     WHERE (mosaic_product_oid={mosaic_product_oid})
#     """
#     if len(auxillary_products) > 1:
#         query_str += f"AND (product_type_sas IN {tuple(auxillary_products)})"
#     elif len(auxillary_products) == 1:
#         query_str += f"AND (product_type_sas='{auxillary_products[0]}')"

#     df = Euclid.launch_job(query_str).get_results().to_pandas()


#     """
#     Can sometimes have multiple auxillary tiles with the same mosaic_product_oid
#     EUC_MER_BGSUB-MOSAIC-VIS_TILE102159774-3EAE6B_20240707T183311.123620Z_00.00.fits
#     EUC_MER_BGSUB-MOSAIC-VIS_TILE102159774-FE2962_20240806T043542.352405Z_00.00.fits
#     For now, take the most recent one
#     """
#     df['creation_date'] = df['file_name'].apply(lambda x: x.split('_')[-2])  # str, lead by the datetime
#     df['tile_index'] = df['file_name'].apply(lambda x: x.split('TILE')[1].split('-')[0])  
#     df = df.sort_values(by='creation_date', ascending=False)  # per tile, newest first
#     df = df.drop_duplicates(subset=['tile_index', 'product_type_sas'], keep='first').reset_index(drop=True)
#     # logging.info(df.iloc[0])
#     return df