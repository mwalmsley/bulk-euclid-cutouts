"""

Designed for bulk (10k+) download of cutouts and PSF from a list of targets known from external catalogs

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF
"""

import logging
import os
import shutil

import healpy
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

from bulk_euclid.utils import pipeline_utils, cutout_utils

# create_folders can be shared

# naming of cutouts should be specific to input style

# choosing of tiles is specific to input style (either all, or specified/read)

# loading and saving a list of coordinates from a tile should be shared


def run(cfg: OmegaConf):
    """
    Convenient wrapper sticking together the steps of the external targets pipeline
    Useful when running from terminal or a script
    See run_from_console.py

    Args:
        cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
    """
    logging.info("Starting external targets pipeline")

    pipeline_utils.create_folders(cfg)

    required_cols = ['id_str', 'target_ra', 'target_dec', 'target_field_of_view', 'category']
    external_targets = pd.read_csv(cfg.external_targets_loc)
    assert all(col in external_targets.columns for col in required_cols), "Missing required columns in external_targets"

    logging.info(external_targets['category'].value_counts())


    # is tile_index is included in external targets csv, use those tiles
    if 'tile_index' in external_targets.columns:
        logging.info('Using tile_index from external targets')
        # this will be used to look up which tiles to use
        # e.g. {'tile_index': [1, 2, 3], 'id_str': ['target1', 'target2', 'target3'], ...}
        targets_with_tiles = external_targets.dropna(subset=['tile_index'])
        assert len(targets_with_tiles) > 0, "No targets with tile_index, likely a bug"
    else:
        # otherwise, look up which tiles to use via healpix MER tile/coordinate file
        logging.info('No tile_index in external targets, looking up tiles by coordinates')

        targets_with_tiles = find_matching_tiles(
            cfg, external_targets
        )  

    logging.info('{} unique tiles for {} targets'.format(targets_with_tiles['tile_index'].nunique(), len(targets_with_tiles)))
    logging.info(targets_with_tiles['category'].value_counts())
    targets_with_tiles.to_csv(cfg.download_dir + '/targets_with_tiles.csv', index=False)

    # targets_with_tiles = targets_with_tiles.sample(2, random_state=42)  # for testing

    

    make_cutouts(cfg, targets_with_tiles)

    make_archive_for_download(cfg)

    logging.info("External targets pipeline complete")
    


def find_matching_tiles(
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

    # logging.info('Loading healpix tile lookup from {}'.format(cfg.healpix_loc))
    try:
        healpix_array = healpy.read_map(cfg.healpix_loc, nest=True)
    except FileNotFoundError as e:
        logging.error(f"Could not find healpix file at {cfg.healpix_loc} - download it first from https://euclid.roe.ac.uk/attachments/153460")
        raise e

    external_targets['tile_index'] = get_matching_tile_indices(
        external_targets['target_ra'].values,
        external_targets['target_dec'].values,
        healpix_array
    )

    logging.info(f'Matched {len(external_targets)} targets to {len(external_targets["tile_index"].unique())} tiles')
    targets_with_tiles = external_targets.dropna(subset=['tile_index'])
    logging.info(f'Targets with possible tile matches: {len(targets_with_tiles)}')
    
    assert len(targets_with_tiles) > 0, "No targets within FoV of any tiles, even before selecting this release: likely a bug"

    logging.info('Selecting only targets with tile in current release')
    cfg.max_tiles = 0  # override to ensure we always get every tile in the release
    tile_indices_in_release = pipeline_utils.get_tile_indices_in_release(cfg)
    external_targets = external_targets[external_targets["tile_index"].isin(tile_indices_in_release)]
    logging.info(f'Targets with tile in current release: {len(external_targets)}')
    assert len(external_targets) > 0, "No targets with tile in current release, check your coordinates are in this release"

    # avoid annoying type conversion
    targets_with_tiles["tile_index"] = targets_with_tiles["tile_index"].astype(int)
    # sort and clean up index
    targets_with_tiles = targets_with_tiles.sort_values('tile_index')
    targets_with_tiles = targets_with_tiles.reset_index(drop=True)

    return targets_with_tiles   


def get_matching_tile_indices(ra: np.ndarray, dec: np.ndarray, healpix_array: np.ndarray) -> np.ndarray:
    """
    Prototype function from D. Sluse to get tile index for a given coord based on an healpix_file
    based on https://gitlab.euclid-sgs.uk/PF-MER/MER_DA/-/blob/develop/MER_DA/python/MER_DA/MER_HPObjectSelection.py?ref_type=heads#L170
    Tile info. based on healpix: -tiling v1.2 -
    https://euclid.roe.ac.uk/projects/mer_pf/wiki/Tiling#Healpix-maps-for-Wide-Field-V12-tiling
    Healpix file 1.2 available on the redmine
    Args:
        ra: np.array of ra in deg (likely output of target['target_ra'].values (for one specific target))
        dec: np.array dec in deg (likely output of target['target_deg'].values (for one specific target))
        healpix_array: np.array w. healpix indices = output of `healpy.read_map('tile_index_map.v1.2.fits.gz', nest=True)`

    Returns: np.array - tile indices

    """
    # convert Ra/Dec from [dec] to [rad]
    # !MER uses math.pi instead np.pi
    #hp_theta = np.pi/2. - dec/180.*np.pi
    hp_theta = np.pi/2. - dec/180.*np.pi
    hp_phi   = ra/180.*np.pi

    # get the healpix indices for all objects
    moc_order = 13    # /!\ HARD-CODED BUT there is a setup in MER code; Should remain 13 forever according to M.Kuemmel
    hp_object_indices = healpy.pixelfunc.ang2pix(healpy.order2nside(moc_order), hp_theta, hp_phi, nest=True)

    tile_index: np.ndarray = healpix_array[hp_object_indices]

    # sometimes the tile index is big endian and datalabs is little endian, byte swap to ensure little endian
    if tile_index.dtype == np.dtype('>i4'):  # big endian int32
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.newbyteorder.html
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html
        # byteswap moves the bytes
        # newbyteorder changes the definition of byte order
        if int(np.__version__[0]) < 2:
            tile_index = tile_index.newbyteorder().byteswap()  # now little endian ; np < 2.0
        else: 
            tile_index = tile_index.view(tile_index.dtype.newbyteorder('=')).byteswap()  # Numpy >= 2.0 

    return tile_index


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
    logging.info('Data products requested: {}'.format(cfg.data_products))

    for tile_n, tile_index in enumerate(unique_tiles):
        logging.info(f'Tile {tile_index}, {tile_n} of {len(unique_tiles)}')
        try:
            
            tile = pipeline_utils.create_tile_object(cfg, tile_index)

            targets_at_that_index = targets_with_tiles.query(f"tile_index == {tile_index}").reset_index(drop=True)

            save_cutouts_for_all_targets_in_that_tile(
                cfg, tile, targets_at_that_index
            )

        except AssertionError as e:
            logging.critical(f"Error downloading tile data and making cutouts for {tile_index}")
            logging.critical(e)






def save_cutouts_for_all_targets_in_that_tile(cfg: OmegaConf, tile: pipeline_utils.Tile, targets_at_that_index: pd.DataFrame) -> None:
    """
    Using the downloaded data products for a single tile listed in dict_of_locs, make cutouts for each target in targets_at_that_index.
    targets_at_that_index is the galaxies within that single tile.

    This function is a bit awkward because we want to load each band separately (to save RAM), 
    but save the cutouts for each target across all bands (so researchers only need a single file per target).

    Args:
        cfg (OmegaConf): cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
       tile:
        targets_at_that_index (pd.DataFrame): The subset of targets (sources) within that single tile. Columns ["tile_index" (now only one), "id_str", "target_ra", "target_dec", "target_field_of_view", "category"]
    """

    assert targets_at_that_index["tile_index"].nunique() == 1


    cutout_data = {}
    header_data = {}
    for band in cfg.bands:
        # this is easier to load once (per band) and then look up each target...
        cutout_data_for_band, header_data_for_band = get_cutout_data_for_band(
            cfg, 
            getattr(tile, band),   # i.e. tile."band", e.g. tile.VIS
            targets_at_that_index
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
            cfg.jpg_dir, str(target["category"]), 'generic', str(target["id_str"]) + "_generic.jpg"  # generic will be replaced
        )
        try:
            if cfg.fits_outputs:
                logging.debug('Saving fits for single galaxy')
                save_multifits_cutout(cfg, target_data, target_header_data, fits_save_loc)
            if cfg.jpg_outputs:
                logging.debug('Saving jpg for single galaxy')
                save_jpg_cutout(cfg, target_data, jpg_save_loc)
        except AssertionError as e:
            logging.critical(f"Error saving cutout for target {target['id_str']}")
            logging.critical(e)
    logging.info('Saved cutouts for all targets in tile {}'.format(target["tile_index"]))


def get_cutout_data_for_band(cfg: OmegaConf, observation: pipeline_utils.Observation, targets_at_that_index: pd.DataFrame) -> dict:
    """
    For a single band, create (in memory) Cutout2D instances for each target in targets_at_that_index using the downloaded data products in dict_of_locs_for_band.
    These Cutout2D instances are later saved as FITS cutouts, but here we just return them.

    The Cutout2D instances are stored in a dict, keyed by the product type (e.g. "FLUX", "MERPSF", "MERRMS", "MERBKG").

    targets_at_that_index is the targets within a single tile.
    
    We load the cutout data band-by-band to avoid blowing up our memory requirements by loading multiple bands at once.
    (hence the awkward footwork with dict_of_locs_for_band)

    The products loaded into cutouts is selected according to cfg.auillary_products.

    Note: the downloaded PSF file contains:
        - an image with PSF cutouts of selected objects arranged next to each other. The stamp pixel size can be found in the header keyword STMPSIZE (e.g. 19 for VIS, 33 for NIR).
        - a table giving the match between the PSF cutout center position (columns x_center and y_center) on the PSF grid image and the coordinate in pixels (columns x and y) or on the sky (Ra, Dec) on the MER tile data.
        https://euclid.roe.ac.uk/issues/22495

    Args:
        cfg (OmegaConf): cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
        observation (Observation): like e.g. {'FLUX': path.fits, 'MERPSF': path.fits, 'MERRMS': path.fits, 'MERBKG': path.fits}
        targets_at_that_index (pd.DataFrame): the targets within a single tile.

    Returns:
        list: of dicts, one per target. Each dict has keys like "BGSUB", "MERPSF", "MERRMS", "MERBKG", and values of Cutout2D instances.
    """
    flux_data = observation.BGSUB.data
    flux_header = observation.BGSUB.header
    flux_wcs = WCS(flux_header)

    if "RMS" in cfg.data_products:
        rms_data = observation.RMS.data
        rms_header = observation.RMS.header
        rms_wcs = WCS(rms_header)

    if "BGMOD" in cfg.data_products:
        bkg_data = observation.BGMOD.data
        bkg_header = observation.BGMOD.header
        bkg_wcs = WCS(bkg_header)

    if "PSF" in cfg.data_products:
        # this is fiddlier due to multi extensions, do manually for now
        psf_loc = observation.PSF.path
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
        # logging.info(target_coord)
        # logging.info(target['target_field_of_view'])
        flux_cutout = Cutout2D(
            data=flux_data,
            position=target_coord,
            # position=target_pixels,
            size=target["target_field_of_view"] * u.arcsec,
            wcs=flux_wcs,
            mode="partial",
        )
        cutout_data_for_target["BGSUB"] = flux_cutout
        header_data_for_target["BGSUB"] = flux_header
        header_data_for_target["BGSUB"]['TARGETX'] = flux_cutout.input_position_cutout[0]
        header_data_for_target["BGSUB"]['TARGETY'] = flux_cutout.input_position_cutout[1]


        if "RMS" in cfg.data_products:
            rms_cutout = Cutout2D(
                data=rms_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=rms_wcs,
                mode="partial",
            )
            cutout_data_for_target["RMS"] = rms_cutout
            header_data_for_target["RMS"] = rms_header
            header_data_for_target["RMS"]['TARGETX'] = rms_cutout.input_position_cutout[0]
            header_data_for_target["RMS"]['TARGETY'] = rms_cutout.input_position_cutout[1]

        if "BGMOD" in cfg.data_products:
            bkg_cutout = Cutout2D(
                data=bkg_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=bkg_wcs,
                mode="partial",
            )
            cutout_data_for_target["BGMOD"] = bkg_cutout
            header_data_for_target["BGMOD"] = bkg_header
            header_data_for_target["BGMOD"]['TARGETX'] = bkg_cutout.input_position_cutout[0]
            header_data_for_target["BGMOD"]['TARGETY'] = bkg_cutout.input_position_cutout[1]

        if "PSF" in cfg.data_products:
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
            )
            cutout_data_for_target["PSF"] = psf_cutout.data
            header_data_for_target["PSF"] = psf_header
            header_data_for_target["PSF"]['TARGETX'] = psf_cutout.input_position_cutout[0]
            header_data_for_target["PSF"]['TARGETY'] = psf_cutout.input_position_cutout[1]

        cutout_data.append(cutout_data_for_target)
        header_data.append(header_data_for_target)

    logging.debug(f'Cutouts made for all targets in band')
    return cutout_data, header_data


def save_jpg_cutout(cfg: OmegaConf, target_data: dict, save_loc: str):

    # makedirs later instead
    # if not os.path.isdir(os.path.dirname(save_loc)):
    #     os.mkdir(os.path.dirname(save_loc))

    assert 'VIS' in target_data.keys()
    vis_im: np.ndarray = target_data['VIS']['BGSUB'].data

    if 'NIR_Y' in target_data.keys():
        y_im: np.ndarray = target_data['NIR_Y']['BGSUB'].data
    else:
        y_im = None

    if 'NIR_J' in target_data.keys():
        j_im: np.ndarray = target_data['NIR_J']['BGSUB'].data
    else:
        j_im = None

    if cfg.add_bkg:
        vis_im = vis_im + target_data['VIS']['BGMOD'].data
        if y_im is not None:
            y_im = y_im + target_data['NIR_Y']['BGMOD'].data
        if j_im is not None:
            j_im = j_im + target_data['NIR_J']['BGMOD'].data

    expected_save_locs = [save_loc.replace('generic', output_format) for output_format in cfg.jpg_outputs]
    logging.debug(expected_save_locs)
    if all([os.path.isfile(loc) for loc in expected_save_locs]) and not cfg.overwrite_jpg:
        logging.debug(f"All jpg already exist for this galaxy, skipping: {save_loc}")
        return

    cutout_utils.save_jpg_cutouts(cfg, save_loc, vis_im, y_im, j_im)





def save_multifits_cutout(cfg: OmegaConf, target_data: dict, target_header_data: dict, save_loc: str):
    """
    Save a list of Cutout2D instances as a FITS file.

    First extension is the empty header, then each subsequent extension is a Cutout2D instance.
    The order is always: BGSUB (flux), PSF, RMS, BGMOD, repeating for each band (ordered like cfg.bands, we suggest sticking to wavelength order)
    MER products not listed in cfg.data_products are not saved.

    By default, the extensions are:

    0: PrimaryHDU (empty)
    1: BGSUB_VIS
    2: PSF_VIS
    3: RMS_VIS
    4: BGMOD_VIS
    5: BGSUB_NIR_Y
    6: PSF_NIR_Y
    7: RMS_NIR_Y
    8: BGMOD_NIR_Y

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
        cutout_flux = band_data["BGSUB"]

        # print(repr(flux_header)) 

        # sanity check
        if np.nanmin(cutout_flux.data) < np.nanmax(cutout_flux.data):
            flux_header = target_header_data[band]["BGSUB"]
            flux_header.update(cutout_flux.wcs.to_header())
            flux_header.append(
                    ("FILTER", band, "Euclid filter for flux image"),
                    end=True,
                )
            flux_hdu = fits.ImageHDU(
                data=cutout_flux.data, name=f"{band}_BGSUB", header=flux_header
            )

        else:
            logging.warning(f"{os.path.basename(save_loc)}: Flux in {band} data is empty, likely a SAS error - saving anyway")
            flux_header = fits.Header()
            flux_hdu = fits.ImageHDU()


        hdu_list.append(flux_hdu)
        # and update the primary header
        header_hdu.header.append(
            (
                f"EXT_{which_extension}",
                f"{band}_BGSUB",
                f"Extension name for {band} BGSUB",
            ),
            end=True,
        )
        which_extension +=1
        
        # TODO this is a bit lazy/repetitive, could be refactored

        if "PSF" in cfg.data_products:
            cutout_psf = band_data["PSF"]
            # psf_header = cutout_psf.wcs.to_header()

            if cutout_psf.min() < cutout_psf.max():

                psf_header = fits.Header()  # blank, always ignored
                psf_header.append(
                    (
                        "FILTER",
                        band,
                        "Euclid filter for PSF image",
                    ),
                    end=True,
                )
                psf_hdu = fits.ImageHDU(
                    data=cutout_psf, name=band+"_PSF", header=psf_header  # NOT .data any more
                )

            else:
                logging.warning(f"{os.path.basename(save_loc)}: PSF in {band} data is empty, likely a SAS error - saving anyway")
                psf_header = fits.Header()
                psf_hdu = fits.ImageHDU()

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
            

        if "RMS" in cfg.data_products:
            cutout_rms = band_data["RMS"]
            if cutout_rms.data.min() < cutout_rms.data.max():
                rms_header = target_header_data[band]["RMS"]
                rms_header.update(cutout_rms.wcs.to_header())
                rms_header.append(
                    (
                        "FILTER",
                        band,
                        "Euclid filter for RMS image",
                    ),
                    end=True,
                )
                rms_hdu = fits.ImageHDU(data=cutout_rms.data, name=band+"_RMS") # TODO changed
            else:
                logging.warning(f"{os.path.basename(save_loc)}: RMS in {band} data is empty, likely a SAS error")
                rms_header = fits.Header()
                rms_hdu = fits.ImageHDU()

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

        if "BGMOD" in cfg.data_products:
            cutout_bkg = band_data["BGMOD"]
            if cutout_bkg.data.min() < cutout_bkg.data.max():

                bkg_header = target_header_data[band]["BGMOD"]
                bkg_header.update(cutout_bkg.wcs.to_header())
                bkg_header.append(
                    (
                        "FILTER",
                        band,
                        "Euclid filter for BKG image",
                    ),
                    end=True,
                )
            
                bkg_hdu = fits.ImageHDU(data=cutout_bkg.data, name=band+"_BKG")

            else:
                logging.warning(f"{os.path.basename(save_loc)}: BKG in {band} data is empty, likely a SAS error")
                bkg_header = fits.Header()
                bkg_hdu = fits.ImageHDU()

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


# def create_folders(cfg: OmegaConf):
#     cfg.download_dir = cfg.base_dir + "/" + cfg.name
#     cfg.tile_dir = cfg.download_dir + "/tiles"

#     cfg.cutout_dir = cfg.download_dir + "/cutouts"
#     cfg.fits_dir = cfg.cutout_dir + "/fits"
#     cfg.jpg_dir = cfg.cutout_dir + "/jpg"

#     cfg.sanity_dir = cfg.download_dir + "/sanity"

#     logging.info(f"Saving to {cfg.download_dir}")
#     assert os.path.exists(os.path.dirname(cfg.download_dir))
#     for d in [
#         cfg.download_dir,
#         cfg.tile_dir,
#         cfg.cutout_dir,
#         cfg.fits_dir,
#         cfg.jpg_dir,
#         cfg.sanity_dir,
#     ]:
#         if not os.path.exists(d):
#             os.makedirs(d)

#     return cfg

def make_archive_for_download(cfg: OmegaConf):
    # list subdirectories within cfg.jpg_dir
    categories = [f for f in os.listdir(cfg.jpg_dir) if os.path.isdir(os.path.join(cfg.jpg_dir, f))]

    logging.info('Archiving cutouts')
    if cfg.jpg_outputs:
        for category in categories:
            # .../cutouts/jpg/known_lens_candidate/sw_arcsinh_vis_only/EUCLJ095929.92+021352.1_sw_arcsinh_vis_only.jpg
            shutil.make_archive(cfg.cutout_dir + f'_jpg_{category}', 'tar', root_dir=cfg.jpg_dir + '/' + category)
            logging.info(f'Archived {category} jpg cutouts')
        logging.info('Archived all jpg cutouts')
    if cfg.fits_outputs:
        for category in categories:
            shutil.make_archive(cfg.cutout_dir + f'_fits_{category}', 'tar', root_dir=cfg.fits_dir + '/' + category)
            logging.info(f'Archived {category} fits cutouts')
        logging.info('Archived fits cutouts')



# def download_all_data_at_tile_index(cfg: OmegaConf, tile_index: int) -> dict:
#     """
#     Download all relevant products for a given tile, including flux data and auxillary data (following cfg.auxillary_data).
#     Returns a dict of paths to each downloaded product, structured like
#     {
#         'VIS': {
#             'FLUX': '{cfg.tile_dir}/EUC_MER_BGSUB-MOSAIC-VIS_TILE...fits',
#             'MERPSF': '{cfg.tile_dir}/EUC_MER_CATALOG-PSF-VIS_TILE...fits',
#             'MERBKG': '{cfg.tile_dir}/EUC_MER_BGMOD-VIS_TILE...fits',
#             'MERRMS': '{cfg.tile_dir}/EUC_MER_MOSAIC-VIS-RMS_TILE...fits'
#         },
#     {
#         'NIR_Y': {
#             'FLUX': '{cfg.tile_dir}/EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE...fits',
#             'MERRMS': '{cfg.tile_dir}/EUC_MER_MOSAIC-NIR-Y-RMS_TILE...fits',
#             'MERPSF': '{cfg.tile_dir}/EUC_MER_CATALOG-PSF-NIR-Y_TILE...fits',
#             'MERBKG': '{cfg.tile_dir}/EUC_MER_BGMOD-NIR-Y_TILE...fits'
#         },
#     }
#     This dict can then be used to make cutouts for each target in that tile.

#     Note:
#     A tile is an area of sky
#     Each tile is identified by a unique tile_index
#     Each tile has many data products associated with it, including the MER mosaic (flux) and auxillary data (PSF, RMS, BKG, etc).

#     Args:
#         cfg (OmegaConf): dictlike with configuration options (folders, bands, auxillary products, etc)
#         tile_index (int): unique identifier of each Euclid tile (sky area). Will download products for this tile.

#     Returns:
#         dict: nested dict of paths to each downloaded product. Structure in docstring above.
#     """
#     flux_tile_metadata = pipeline_utils.get_tiles_in_survey(
#         tile_index=tile_index, bands=cfg.bands, release_name=cfg.release_name
#     )

#     dict_of_locs = {}

#     # download all the flux tiles with that index
#     if cfg.download_method == 'datalabs_path':
#         flux_tile_metadata['file_loc'] = flux_tile_metadata['datalabs_path'] + '/' + flux_tile_metadata['file_name']
#     else:
#         flux_tile_metadata = pipeline_utils.save_euclid_products(
#             flux_tile_metadata, download_dir=cfg.tile_dir
#         )
#     for _, flux_tile in flux_tile_metadata.iterrows():
#         dict_of_locs[flux_tile["filter_name"]] = {"FLUX": flux_tile["file_loc"]}  # will add other keys laters

#     if cfg.add_bkg and 'MERBKG' not in cfg.auxillary_products:
#         cfg.auxillary_products.append('MERBKG')  # add BKG if not already requested

#     # also download all auxillary data for that tile
#     if cfg.auxillary_products == []:
#         logging.info('No auxillary data requested, only downloading flux')
#         these_aux_locs = {}
#     else:
#         logging.info('Downloading auxillary data for tile {}'.format(tile_index))
#         for _, flux_tile in flux_tile_metadata.iterrows():
#             # could have used tile_index for this search, but we want to restrict to some bands only
#             auxillary_tile_metadata = pipeline_utils.get_auxillary_tiles(
#                 flux_tile["mosaic_product_oid"], auxillary_products=cfg.auxillary_products
#             )
#             if cfg.download_method == 'datalabs_path':
#                 auxillary_tile_metadata['file_loc'] = auxillary_tile_metadata['datalabs_path'] + '/' + auxillary_tile_metadata['file_name']
#             else:
#                 auxillary_tile_metadata = pipeline_utils.save_euclid_products(
#                     auxillary_tile_metadata, download_dir=cfg.tile_dir
#                 )
#             these_aux_locs = dict(
#                 zip(
#                     auxillary_tile_metadata["product_type_sas"],  # e.g. MERPSF
#                     auxillary_tile_metadata["file_loc"],  # path to downloaded file
#                 )
#             )
#             # add tracking of the auxillary data to existing dict, previously with only FLUX key
#             # now like {FLUX: path, MERPSF: path, MERRMS: path, MERBKG: path}
#             dict_of_locs[flux_tile["filter_name"]].update(**these_aux_locs) 

#     logging.debug(f"Downloaded flux+auxillary tiles: {dict_of_locs}")
#     logging.info('Downloaded all data for tile {}'.format(tile_index))
#     # assert len(dict_of_locs.keys()) == cfg.bands, f"Missing bands in downloaded data: {len(dict_of_locs.keys())} of {len(cfg.bands)} keys, {dict_of_locs.keys()} vs {cfg.bands}"
#     assert set(cfg.bands) == set(dict_of_locs.keys()), f'Downloaded bands dont match expected bands: downloaded {set(dict_of_locs.keys())}, expected {set(cfg.bands)}'
#     return dict_of_locs


# def cutout_psf_manually(psf_grid, x_center, y_center, cutout_size):
#     #cutout is the size of the image cutout to search for the PSFs in that space
#     x_start = int(round(x_center - cutout_size / 2))
#     x_end = x_start + cutout_size
#     y_start = int(round(y_center - cutout_size / 2))
#     y_end = y_start + cutout_size

#     # avoid edge effects (possibly not needed)
#     if x_start < 0:
#         x_start = 0
#     if x_end > psf_grid.shape[1]:
#         x_end = psf_grid.shape[1]
#     if y_start < 0:
#         y_start = 0
#     if y_end > psf_grid.shape[0]:
#         y_end = psf_grid.shape[0]
#     # logging.debug(f'before edge: {y_start} {y_end}, {x_start} {x_end}')

#     # make the slice

#     # logging.debug(f'first: {y_start} {y_end}, {x_start} {x_end}')
#     cutout = psf_grid[y_start:y_end, x_start:x_end]

#     # find the maxima
#     max_y_local, max_x_local = np.unravel_index(np.argmax(cutout), cutout.shape)
#     max_x_global = x_start + max_x_local
#     max_y_global = y_start + max_y_local
#     brightest_pixels = [max_x_global, max_y_global]

#     # update x_center and y_center to the actual brightest pixels
#     x_center = brightest_pixels[0]
#     y_center = brightest_pixels[1]

#     # make the slice AGAIN
#     x_start = int(x_center - cutout_size / 2)
#     x_end = int(x_center + cutout_size / 2)
#     y_start = int(y_center - cutout_size / 2)
#     y_end = int(y_center + cutout_size / 2)
#     # logging.debug(f'second: {y_start} {y_end}, {x_start} {x_end}')
#     cutout = psf_grid[y_start:y_end, x_start:x_end]

#     return cutout
