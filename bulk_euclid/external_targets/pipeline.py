"""

Designed for bulk (10k+) download of cutouts and PSF from a list of targets known from external catalogs

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF
"""

import logging
# import warnings
import os

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.neighbors import KDTree
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.table import Table
# from astropy.io.fits.verify import VerifyWarning
from PIL import Image

from bulk_euclid.utils import pipeline_utils, cutout_utils


def run(cfg: OmegaConf):
    """
    Convenient wrapper sticking together the steps of the external targets pipeline
    Useful when running from terminal or a script
    See run_from_console.py

    Args:
        cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
    """
    logging.info("Starting external targets pipeline")

    create_folders(cfg)
    pipeline_utils.login()

    external_targets = pd.read_csv(cfg.external_targets_loc)


    # NOW we go!
    # matching each target with the best tile
    targets_with_tiles = get_matching_tiles(
        cfg, external_targets
    )  
    logging.info('Targets per release: \n{}'.format(targets_with_tiles['release_name'].value_counts()))
    logging.info('{} unqiue tiles for {} targets'.format(targets_with_tiles['tile_index'].nunique(), len(targets_with_tiles)))

    make_cutouts(cfg, targets_with_tiles)

    logging.info("External targets pipeline complete")


def get_matching_tiles(
    cfg: OmegaConf, external_targets: pd.DataFrame = None
):  # simplified from a_make_catalogs_and_cutouts.py
    """
    For each target in the external_targets dataframe, find the closest tile that covers it.
    It returns a dataframe acting as a lookup table between target and tile.
    This is then used later to choose which tiles to download and, for each tile, which targets to make cutouts of

    external_targets must have columns ['id_str', 'target_ra' (deg), 'target_dec' (deg), 'target_field_of_view' (arcsec)]. 
    id_str has no effect, it's just a primary key.

    Args:
        cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        external_targets (pd.DataFrame, optional): with columns ['id_str', 'target_ra' (deg), 'target_dec' (deg), 'target_field_of_view' (arcsec)]. id_str has no effect, it's just a primary key. Defaults to None.

    Returns:
        pd.DataFrame: with columns ['tile_index', 'id_str', 'target_ra', 'target_dec', 'target_field_of_view']
    """

    # get all VIS tiles in that release (e.g. F-006 for Wide)
    # we'll get all the other bands and auxillary data later. Here we just need the tile index.
    tiles = pipeline_utils.get_tiles_in_survey(
        bands=["VIS"], release_name=cfg.release_name
    )
    assert len(tiles) > 0
    tiles.to_csv("temp_tiles.csv")  # for debugging

    # add tile FoV
    # adds cols ['ra_min', 'ra_max', 'dec_min', 'dec_max'] by unpacking the "fov" tile metadata column
    tiles = pipeline_utils.get_tile_extents_fov(tiles)

    # use KDTree to quickly find the closest tile to each target
    tile_kdtree = KDTree(tiles[["ra", "dec"]].values)  # coordinates of the tile centers

    # e.g. ['CALBLOCK_PV-005_R2', 'CALBLOCK_PV-005_R3', 'F-003_240321', 'F-003_240612' , 'F-006']
    if cfg.release_priority is None:
        release_priority_key = None
    else:
        release_priority_key = dict(zip(cfg.release_priority, range(len(cfg.release_priority))))
        # e.g. {'CALBLOCK_PV-005_R2': 0, 'CALBLOCK_PV-005_R3': 1, ...}

    logging.info('Begin target/tile cross-match')
    for target_n, target in external_targets.iterrows():
        # query for all tiles within one degree of target
        close_tile_indices = tile_kdtree.query_radius(target[['target_ra', 'target_dec']].values.reshape(1, -1), r=1)[0]  # 0 for first row, all results
        close_tiles = tiles.iloc[close_tile_indices]

        if len(close_tiles) > 0:

            safety_margin = 0.01 # degrees

            # check which close tiles are actually within the FoV
            # this will fail for tiles on the RA flip boundary, but none yet TODO
            within_ra = (
                close_tiles["ra_min"] + safety_margin < target["target_ra"] ) & (
                target["target_ra"] < close_tiles["ra_max"] - safety_margin
            )
            within_dec = (
                close_tiles["dec_min"] + safety_margin < target["target_dec"] ) & (
                target["target_dec"] < close_tiles["dec_max"] - safety_margin
            )
            close_tiles = close_tiles[within_ra & within_dec]

            if len(close_tiles) > 0:
                # we have at least one tile within the FoV, which should we use?

                if release_priority_key is None:  
                    # user did not set a priority order for the tiles in cfg.release_priority
                    # just pick the first tile that's within the FoV
                    chosen_tile = close_tiles.iloc[0]
                else:
                    # pick in order of release priority
                    # if the release is not recognised, it gets a priority of -1 (lowest)
                    # higher priority is a higher number (higher index in cfg.release_priority)
                    # after sorting (ascending) by release priority, pick the last one for tile with the highest priority
                    close_tiles['priority'] = close_tiles['release_name'].apply(lambda x: release_priority_key.get(x, -1))
                    chosen_tile = close_tiles.sort_values(by='priority').iloc[-1]
                external_targets.loc[target_n, "tile_index"] = chosen_tile["tile_index"]
                # useful for debugging
                external_targets.loc[target_n, "release_name"] = chosen_tile['release_name']
                external_targets.loc[target_n, "tile_ra"] = chosen_tile['ra']
                external_targets.loc[target_n, "tile_ra_min"] = chosen_tile['ra_min']
                external_targets.loc[target_n, "tile_ra_max"] = chosen_tile['ra_max']
                external_targets.loc[target_n, "tile_dec_min"] = chosen_tile['dec_min']
                external_targets.loc[target_n, "tile_dec_max"] = chosen_tile['dec_max']
                external_targets.loc[target_n, "tile_dec"] = chosen_tile['dec']


    logging.info(f'Matched {len(external_targets)} targets to {len(external_targets["tile_index"].unique())} tiles')
    targets_with_tiles = external_targets.dropna(subset=['tile_index'])
    logging.info(f'Targets with tile matches: {len(targets_with_tiles)}')
    
    assert len(targets_with_tiles) > 0, "No targets within FoV of any tiles, likely a bug"
    assert len(targets_with_tiles) > 0

    # avoid annoying type conversion
    targets_with_tiles["tile_index"] = targets_with_tiles["tile_index"].astype(int)
    # clean up index
    targets_with_tiles = targets_with_tiles.reset_index(drop=True)

    return targets_with_tiles


def make_cutouts(cfg: OmegaConf, targets_with_tiles: pd.DataFrame) -> None:
    """
    For each tile in targets_with_tiles, download all the data for that tile, and make cutouts for each target within that tile.
    The cutouts will include flux data and auxillary data (PSF, RMS, BKG) if requested in cfg.auxillary_products

    targets_with_tiles must have columns ['tile_index', 'id_str', 'target_ra', 'target_dec', 'target_field_of_view']

    Args:
        cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        targets_with_tiles (pd.DataFrame): Lookup table linking each target with the tile covering that target. 

    Raises:
        e: Download error (e.g. when SAS is temporarily down)
    """
    unique_tiles = targets_with_tiles["tile_index"].unique()
    for tile_n, tile_index in enumerate(unique_tiles):
        logging.info(f'Tile {tile_index}, {tile_n} of {len(unique_tiles)}')
        try:
            dict_of_locs = download_all_data_at_tile_index(cfg, tile_index)
            logging.debug(f"Downloaded: {dict_of_locs}")

            targets_at_that_index = targets_with_tiles.query(f"tile_index == {tile_index}").reset_index(drop=True)

            save_cutouts_for_all_targets_in_that_tile(
                cfg, dict_of_locs, targets_at_that_index
            )

        except AssertionError as e:
            logging.critical(f"Error downloading tile data and making cutouts for {tile_index}")
            logging.critical(e)

        if cfg.delete_tiles:
            try:
                logging.info('Deleting tile')
                for band in cfg.bands:
                    os.remove(dict_of_locs[band]["FLUX"])
                    for aux in cfg.auxillary_products:
                        os.remove(dict_of_locs[band][aux])
            except Exception as e:
                logging.error(f"Error deleting tile {tile_index}")
                logging.error(e)



def download_all_data_at_tile_index(cfg: OmegaConf, tile_index: int) -> dict:
    """
    Download all relevant products for a given tile, including flux data and auxillary data (following cfg.auxillary_data).
    Returns a dict of paths to each downloaded product, structured like
    {
        'VIS': {
            'FLUX': '{cfg.tile_dir}/EUC_MER_BGSUB-MOSAIC-VIS_TILE...fits',
            'MERPSF': '{cfg.tile_dir}/EUC_MER_CATALOG-PSF-VIS_TILE...fits',
            'MERBKG': '{cfg.tile_dir}/EUC_MER_BGMOD-VIS_TILE...fits',
            'MERRMS': '{cfg.tile_dir}/EUC_MER_MOSAIC-VIS-RMS_TILE...fits'
        },
    {
        'NIR_Y': {
            'FLUX': '{cfg.tile_dir}/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE...fits',
            'MERRMS': '{cfg.tile_dir}/EUC_MER_MOSAIC-NIR-Y-RMS_TILE...fits',
            'MERPSF': '{cfg.tile_dir}/EUC_MER_CATALOG-PSF-NIR-Y_TILE...fits',
            'MERBKG': '{cfg.tile_dir}/EUC_MER_BGMOD-NIR-Y_TILE...fits'
        },
    }
    This dict can then be used to make cutouts for each target in that tile.

    Note:
    A tile is an area of sky
    Each tile is identified by a unique tile_index
    Each tile has many data products associated with it, including the MER mosaic (flux) and auxillary data (PSF, RMS, BKG, etc).

    Args:
        cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        tile_index (int): unique identifier of each Euclid tile (sky area). Will download products for this tile.

    Returns:
        dict: nested dict of paths to each downloaded product. Structure in docstring above.
    """
    flux_tile_metadata = pipeline_utils.get_tiles_in_survey(
        tile_index=tile_index, bands=cfg.bands, release_name=cfg.release_name
    )
    # download all the flux tiles with that index
    flux_tile_metadata = pipeline_utils.save_euclid_products(
        flux_tile_metadata, download_dir=cfg.tile_dir
    )

    dict_of_locs = {}

    # download all auxillary data for that tile
    logging.info('Downloading all data for tile {}'.format(tile_index))
    for _, flux_tile in flux_tile_metadata.iterrows():
        # could have used tile_index for this search, but we want to restrict to some bands only
        auxillary_tile_metadata = pipeline_utils.get_auxillary_tiles(
            flux_tile["mosaic_product_oid"], auxillary_products=cfg.auxillary_products
        )
        auxillary_tile_metadata = pipeline_utils.save_euclid_products(
            auxillary_tile_metadata, download_dir=cfg.tile_dir
        )
        these_aux_locs = dict(
            zip(
                auxillary_tile_metadata["product_type_sas"],
                auxillary_tile_metadata["file_loc"],
            )
        )
        dict_of_locs[flux_tile["filter_name"]] = {
            "FLUX": flux_tile["file_loc"],
            **these_aux_locs,
        }

    logging.debug(f"Downloaded flux+auxillary tiles: {dict_of_locs}")
    logging.info('Downloaded all data for tile {}'.format(tile_index))
    # assert len(dict_of_locs.keys()) == cfg.bands, f"Missing bands in downloaded data: {len(dict_of_locs.keys())} of {len(cfg.bands)} keys, {dict_of_locs.keys()} vs {cfg.bands}"
    assert set(cfg.bands) == set(dict_of_locs.keys()), f'Downloaded bands dont match expected bands: downloaded {set(dict_of_locs.keys())}, expected {set(cfg.bands)}'
    return dict_of_locs


def save_cutouts_for_all_targets_in_that_tile(cfg: OmegaConf, dict_of_locs: dict, targets_at_that_index: pd.DataFrame) -> None:
    """
    Using the downloaded data products for a single tile listed in dict_of_locs, make cutouts for each target in targets_at_that_index.
    targets_at_that_index is the galaxies within that single tile.

    This function is a bit awkward because we want to load each band separately (to save RAM), 
    but save the cutouts for each target across all bands (so researchers only need a single file per target).

    Args:
        cfg (OmegaConf): cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        dict_of_locs (dict): nested dict of paths to (already downloaded) products for each band, for a single tile. Structure in docstring of download_all_data_at_tile_index
        targets_at_that_index (pd.DataFrame): The subset of targets (sources) within that single tile. Columns ["tile_index" (now only one), "id_str", "target_ra", "target_dec", "target_field_of_view", "category"]
    """

    assert targets_at_that_index["tile_index"].nunique() == 1

    cutout_data = {}
    header_data = {}
    for band in cfg.bands:
        # this is easier to load once (per band) and then look up each target...
        cutout_data_for_band, header_data_for_band = get_cutout_data_for_band(
            cfg, dict_of_locs[band], targets_at_that_index
        )
        cutout_data[band] = cutout_data_for_band
        header_data[band] = header_data_for_band
        logging.info('Cutout data sliced for band {}'.format(band))
        # so each cutout_data[band] is a list of dicts, one per target, like [{'FLUX': flux_cutout, 'MERPSF': psf_cutout, ...}, ...]
    # ...but saving fits we want to iterate over targets first, and get the data across all bands
    logging.info('Cutout data sliced for all bands, begin saving to disk')
    for target_n, target in targets_at_that_index.iterrows():
        # this reshapes the data to be a nested dict, with the top level keyed by band, and the inner level keyed by product type (exactly like dict_of_locs)
        # e.g. { VIS: {FLUX: flux_cutout, MERPSF: psf_cutout, ...}, NIR_Y: {...}, ...}
        target_data = { band: cutout_data[band][target_n] for band in cfg.bands }
        target_header_data = { band: header_data[band][target_n] for band in cfg.bands }
        # updated to save by category, assuming less than e.g. 50k targets per category
        fits_save_loc = os.path.join(
            cfg.fits_dir, str(target["category"]), str(target["id_str"]) + ".fits"
        )
        jpg_save_loc = os.path.join(
            cfg.jpg_dir, str(target["category"]), str(target["id_str"]) + ".jpg"
        )
        try:
            if cfg.fits_outputs:
                save_multifits_cutout(cfg, target_data, target_header_data, fits_save_loc)
            if cfg.jpg_outputs:
                save_jpg_cutout(cfg, target_data, jpg_save_loc)
        except AssertionError as e:
            logging.critical(f"Error saving cutout for target {target['id_str']}")
            logging.critical(e)
    logging.info('Saved cutouts for all targets in tile {}'.format(target["tile_index"]))


def get_cutout_data_for_band(cfg: OmegaConf, dict_of_locs_for_band: dict, targets_at_that_index: pd.DataFrame) -> dict:
    """
    For a single band, create (in memory) Cutout2D instances for each target in targets_at_that_index using the downloaded data products in dict_of_locs_for_band.
    These Cutout2D instances are later saved as FITS cutouts, but here we just return them.

    The Cutout2D instances are stored in a dict, keyed by the product type (e.g. "FLUX", "MERPSF", "MERRMS", "MERBKG").

    targets_at_that_index is the targets within a single tile.
    dict_of_locs_for_band is the paths to the downloaded data products for that band, for that tile. It is exactly like dict_of_locs, but keyed into a single band.
    We load the cutout data band-by-band to avoid blowing up our memory requirements by loading multiple bands at once.
    (hence the awkward footwork with dict_of_locs_for_band)

    The products loaded into cutouts is selected according to cfg.auillary_products.

    Note: the downloaded PSF file contains:
        - an image with PSF cutouts of selected objects arranged next to each other. The stamp pixel size can be found in the header keyword STMPSIZE (e.g. 19 for VIS, 33 for NIR).
        - a table giving the match between the PSF cutout center position (columns x_center and y_center) on the PSF grid image and the coordinate in pixels (columns x and y) or on the sky (Ra, Dec) on the MER tile data.
        https://euclid.roe.ac.uk/issues/22495

    Args:
        cfg (OmegaConf): cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        dict_of_locs_for_band (dict): like dict_of_locs above, but for one band. E.g. {'FLUX': path.fits, 'MERPSF': path.fits, 'MERRMS': path.fits, 'MERBKG': path.fits}
        targets_at_that_index (pd.DataFrame): the targets within a single tile.

    Returns:
        list: of dicts, one per target. Each dict has keys like "FLUX", "MERPSF", "MERRMS", "MERBKG", and values of Cutout2D instances.
    """
    flux_data, flux_header = fits.getdata(dict_of_locs_for_band["FLUX"], header=True)
    flux_wcs = WCS(flux_header)

    if "MERRMS" in cfg.auxillary_products:
        rms_loc = dict_of_locs_for_band["MERRMS"]
        rms_data, rms_header = fits.getdata(rms_loc, header=True)
        rms_wcs = WCS(rms_header)

    if "MERBKG" in cfg.auxillary_products:
        bkg_loc = dict_of_locs_for_band["MERBKG"]
        bkg_data, bkg_header = fits.getdata(bkg_loc, header=True)
        bkg_wcs = WCS(bkg_header)

    if "MERPSF" in cfg.auxillary_products:
        psf_loc = dict_of_locs_for_band["MERPSF"]

        psf_tile, psf_header = fits.getdata(psf_loc, ext=1, header=True)
        stamp_size = psf_header["STMPSIZE"]
        psf_table = Table.read(fits.open(psf_loc)[2]).to_pandas()
        psf_tree = KDTree(psf_table[["x", "y"]]) # build tree using x, y, the pixel coordinates of the PSF in the MER tile
        psf_wcs = WCS(psf_header)

    logging.info('Loaded tile, ready to slice')

    cutout_data = []
    header_data = []
    for target_n, target in targets_at_that_index.iterrows():
        logging.debug(f"target {target_n} of {len(targets_at_that_index)}")

        cutout_data_for_target = {}
        header_data_for_target = {}

        # cut out the flux data
        target_coord = SkyCoord(
            target["target_ra"], target["target_dec"], frame="icrs", unit="deg"
        )
        target_pixels = flux_wcs.world_to_pixel(target_coord)
        assert target_pixels[0] > 0 and target_pixels[1] > 0, f"Target {target_n} has negative pixel coordinates, likely a WCS error or target just outside tile: {target_pixels}"
        if target_pixels[0] > 19200 and target_pixels[1] > 19200:
            logging.warning(f"Target {target_n} has too-large pixel coordinates, likely a WCS error or target just outside tile: {target_pixels}")
        # logging.info(target)
        # logging.info('WCS: {}'.format(flux_wcs))
        # logging.info(f"Flux center: {target_coord}")
        # logging.info(f"Flux center pixels: {target_pixels}")
        flux_cutout = Cutout2D(
            data=flux_data,
            position=target_coord,
            # position=target_pixels,
            size=target["target_field_of_view"] * u.arcsec,
            wcs=flux_wcs,
            mode="partial",
        )
        cutout_data_for_target["FLUX"] = flux_cutout
        header_data_for_target["FLUX"] = flux_header
        

        if "MERRMS" in cfg.auxillary_products:
            rms_cutout = Cutout2D(
                data=rms_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=rms_wcs,
                mode="partial",
            )
            cutout_data_for_target["MERRMS"] = rms_cutout
            header_data_for_target["MERRMS"] = rms_header

        if "MERBKG" in cfg.auxillary_products:
            bkg_cutout = Cutout2D(
                data=bkg_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=bkg_wcs,
                mode="partial",
            )
            cutout_data_for_target["MERBKG"] = bkg_cutout
            cutout_data_for_target["MERBKG"] = bkg_header

        if "MERPSF" in cfg.auxillary_products:
            # find pixel coordinates of target in PSF tile
            # now changed to flux WCS as PSF WCS is wrong according to MER
            target_pixels = flux_wcs.world_to_pixel(target_coord)  # the pixel coordinates of the galaxy in MER tile
            # find pixel coordinates of closest PSF to target
            _, psf_index = psf_tree.query(
                np.array(target_pixels).reshape(1, -1), k=1
            )  # single sample reshape
            # TODO add warning if distance is large (the underscore)
            # scalar: 1 search, with 1 neighbour result
            psf_index = psf_index.squeeze()
            # get that PSF row
            closest_psf = psf_table.iloc[psf_index]
            # this is the metadata row describing the PSF with the closest sky coordinates to the target

            # slice out that PSF
            psf_center_pixels = (closest_psf["x_center"]-1, closest_psf["y_center"]-1)

            psf_cutout = Cutout2D(
                data=psf_tile,
                position=psf_center_pixels,  # slice using x_center, y_center, the pixel coordinates of the PSF center in the PSF tile
                size=stamp_size,
                wcs=psf_wcs,
                mode="partial",
            ).data

            cutout_data_for_target["MERPSF"] = psf_cutout
            header_data_for_target["MERPSF"] = psf_header

        cutout_data.append(cutout_data_for_target)
        header_data.append(header_data_for_target)

    logging.debug(f'Cutouts made for all targets in band')
    return cutout_data, header_data


def save_jpg_cutout(cfg: OmegaConf, target_data: dict, save_loc: str):

    if not os.path.isdir(os.path.dirname(save_loc)):
        os.mkdir(os.path.dirname(save_loc))

    assert 'VIS' in target_data.keys()
    vis_im: np.ndarray = target_data['VIS']['FLUX'].data

    if 'NIR_Y' in target_data.keys():
        y_im: np.ndarray = target_data['NIR_Y']['FLUX'].data
    else:
        y_im = None

    if 'NIR_J' in target_data.keys():
        j_im: np.ndarray = target_data['NIR_J']['FLUX'].data
    else:
        j_im = None

    cutout_utils.save_jpg_cutouts(cfg, save_loc, vis_im, y_im, j_im)





def save_multifits_cutout(cfg: OmegaConf, target_data: dict, target_header_data: dict, save_loc: str):
    """
    Save a list of Cutout2D instances as a FITS file.

    First extension is the empty header, then each subsequent extension is a Cutout2D instance.
    The order is always: FLUX, MERPSF, MERRMS, MERBKG, repeating for each band (ordered like cfg.bands, we suggest sticking to wavelength order)
    MER products not listed in cfg.auxillary_products are not saved.

    By default, the extensions are:

    0: PrimaryHDU (empty)
    1: FLUX_VIS
    2: MERPSF_VIS
    3: MERRMS_VIS
    4: MERBKG_VIS
    5: FLUX_NIR_Y
    6: MERPSF_NIR_Y
    7: MERRMS_NIR_Y
    8: MERBKG_NIR_Y

    Each extension has a WCS header, and a FILTER keyword to indicate the band.

    target_data is a nested dict, with the top level keyed by band, and the inner level keyed by product type (exactly like dict_of_locs)
    e.g. { VIS: {FLUX: flux_cutout, MERPSF: psf_cutout, ...}, NIR_Y: {...}, ...}

    Args:
        cfg (OmegaConf): cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        target_data (dict): cutouts for one target, like {'VIS': {'FLUX': flux_cutout, 'MERPSF': psf_cutout, ...}, NIR_Y: {...}, ...}
        save_loc (str): path to save fits file (including .fits extension)
    """

    if os.path.isfile(save_loc) and not cfg.overwrite_fits:
        logging.debug(f"File already exists, skipping: {save_loc}")
        return

    header_hdu = fits.PrimaryHDU()
    which_extension = 1

    hdu_list = [header_hdu]

    for band in cfg.bands:
        band_data = target_data[band]
        cutout_flux = band_data["FLUX"]
        flux_header = target_header_data[band]["FLUX"]
        flux_header.update(cutout_flux.wcs.to_header())
        # flux_header['EXTNAME'] = 'FLUX'
        # flux_header.set('EXTNAME', 'FLUX')
        flux_header.append(
            ("FILTER", band, "Euclid filter for flux image"),
            end=True,
        )
        # print(repr(flux_header)) 

        # sanity check
        # logging.info(cutout_flux.data.shape)
        assert np.nanmin(cutout_flux.data) < np.nanmax(cutout_flux.data), f"{os.path.basename(save_loc)}: Flux in {band} data is empty, likely a SAS error"
        flux_hdu = fits.ImageHDU(
            data=cutout_flux.data, name=f"{band}_FLUX", header=flux_header
        )
        hdu_list.append(flux_hdu)
        # and update the primary header
        header_hdu.header.append(
            (
                f"EXT_{which_extension}",
                f"{band}_FLUX",
                f"Extension name for {band} flux",
            ),
            end=True,
        )
        which_extension +=1

        # TODO this is a bit lazy/repetitive, could be refactored

        if "MERPSF" in cfg.auxillary_products:
            cutout_psf = band_data["MERPSF"]
            # psf_header = cutout_psf.wcs.to_header()
            # psf_header['EXTNAME'] = 'MERPSF'
            psf_header = fits.Header()  # blank, always ignored
            psf_header.append(
                (
                    "FILTER",
                    band,
                    "Euclid filter for PSF image",
                ),
                end=True,
            )
            assert cutout_psf.min() < cutout_psf.max(), f"{os.path.basename(save_loc)}: PSF in {band} data is empty, likely a SAS error"
            psf_hdu = fits.ImageHDU(
                data=cutout_psf, name=band+"_PSF", header=psf_header  # NOT .data any more
            )
            hdu_list.append(psf_hdu)
            header_hdu.header.append(
            (
                f"EXT_{which_extension}",
                f"{band}_PSF",
                f"Extension name for {band} PSF",
            ),
            end=True,
            )
            which_extension +=1

        if "MERRMS" in cfg.auxillary_products:
            cutout_rms = band_data["MERRMS"]
            rms_header = target_header_data[band]["MERRMS"]
            rms_header.update(cutout_rms.wcs.to_header())
            # rms_header['EXTNAME'] = 'MERPSF'
            rms_header.append(
                (
                    "FILTER",
                    band,
                    "Euclid filter for RMS image",
                ),
                end=True,
            )
            # logging.info(cutout_rms.data.shape)
            assert cutout_rms.data.min() < cutout_rms.data.max(), f"{os.path.basename(save_loc)}: RMS in {band} data is empty, likely a SAS error"
            rms_hdu = fits.ImageHDU(data=cutout_rms.data, name=band+"_RMS") # TODO changed
            hdu_list.append(rms_hdu)
            header_hdu.header.append(
            (
                f"EXT_{which_extension}",
                f"{band}_RMS",
                f"Extension name for {band} RMS",
            ),
            end=True,
            )
            which_extension +=1

        if "MERBKG" in cfg.auxillary_products:
            cutout_bkg = band_data["MERBKG"]
            bkg_header = target_header_data[band]["MERBKG"]
            bkg_header.update(cutout_bkg.wcs.to_header())
            # bkg_header['EXTNAME'] = 'MERBKG'
            bkg_header.append(
                (
                    "FILTER",
                    band,
                    "Euclid filter for BKG image",
                ),
                end=True,
            )
            assert cutout_bkg.data.min() < cutout_bkg.data.max(), f"{os.path.basename(save_loc)}: BKG in {band} data is empty, likely a SAS error"
            bkg_hdu = fits.ImageHDU(data=cutout_bkg.data, name=band+"_BKG")
            hdu_list.append(bkg_hdu)
            header_hdu.header.append(
            (
                f"EXT_{which_extension}",
                f"{band}_BKG",
                f"Extension name for {band} BKG",
            ),
            end=True,
            )
            which_extension +=1

    hdul = fits.HDUList(hdu_list)

    if not os.path.isdir(os.path.dirname(save_loc)):
        os.mkdir(os.path.dirname(save_loc))

    hdul.writeto(save_loc, overwrite=True)


def create_folders(cfg: OmegaConf):
    cfg.download_dir = cfg.base_dir + "/" + cfg.name
    cfg.tile_dir = cfg.download_dir + "/tiles"

    cfg.cutout_dir = cfg.download_dir + "/cutouts"
    cfg.fits_dir = cfg.cutout_dir + "/fits"
    cfg.jpg_dir = cfg.cutout_dir + "/jpg"

    cfg.sanity_dir = cfg.download_dir + "/sanity"

    logging.info(f"Saving to {cfg.download_dir}")
    assert os.path.exists(os.path.dirname(cfg.download_dir))
    for d in [
        cfg.download_dir,
        cfg.tile_dir,
        cfg.cutout_dir,
        cfg.fits_dir,
        cfg.jpg_dir,
        cfg.sanity_dir,
    ]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg



def cutout_psf_manually(psf_grid, x_center, y_center, cutout_size):
    #cutout is the size of the image cutout to search for the PSFs in that space
    x_start = int(round(x_center - cutout_size / 2))
    x_end = x_start + cutout_size
    y_start = int(round(y_center - cutout_size / 2))
    y_end = y_start + cutout_size

    # avoid edge effects (possibly not needed)
    if x_start < 0:
        x_start = 0
    if x_end > psf_grid.shape[1]:
        x_end = psf_grid.shape[1]
    if y_start < 0:
        y_start = 0
    if y_end > psf_grid.shape[0]:
        y_end = psf_grid.shape[0]
    # logging.debug(f'before edge: {y_start} {y_end}, {x_start} {x_end}')

    # make the slice

    # logging.debug(f'first: {y_start} {y_end}, {x_start} {x_end}')
    cutout = psf_grid[y_start:y_end, x_start:x_end]

    # find the maxima
    max_y_local, max_x_local = np.unravel_index(np.argmax(cutout), cutout.shape)
    max_x_global = x_start + max_x_local
    max_y_global = y_start + max_y_local
    brightest_pixels = [max_x_global, max_y_global]

    # update x_center and y_center to the actual brightest pixels
    x_center = brightest_pixels[0]
    y_center = brightest_pixels[1]

    # make the slice AGAIN
    x_start = int(x_center - cutout_size / 2)
    x_end = int(x_center + cutout_size / 2)
    y_start = int(y_center - cutout_size / 2)
    y_end = int(y_center + cutout_size / 2)
    # logging.debug(f'second: {y_start} {y_end}, {x_start} {x_end}')
    cutout = psf_grid[y_start:y_end, x_start:x_end]

    return cutout