"""

Designed for bulk (10k+) download of cutouts and PSF from a list of targets known from external catalogs

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF
"""

import logging
import warnings
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
from astropy.io.fits.verify import VerifyWarning

from bulk_euclid.utils import pipeline_utils


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

    # with columns ['id_str', 'target_ra' (deg), 'target_dec' (deg), 'target_field_of_view' (arcsec)].
    external_targets = pd.read_csv(cfg.external_targets_loc)

    targets_with_tiles = get_matching_tiles(
        cfg, external_targets
    )  # matching each target with the best tile

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
    target_coords = external_targets[["target_ra", "target_dec"]].values
    tile_indices = tile_kdtree.query(target_coords, k=1, return_distance=False)
    tile_indices = tile_indices[:, 0]
    # the numeric index (0, 1, ...) of the closest tile to each target
    # (not related to the tile_index column, confusingly)

    # pick out the closest matching tile for each target
    target_tiles = tiles.iloc[tile_indices].copy()
    target_tiles = target_tiles.reset_index(drop=True)  # use the new numeric order
    # stick together
    target_tiles = pd.concat([target_tiles, external_targets], axis=1)

    # check if target is within tile FoV
    within_ra = (target_tiles["ra_min"] < target_tiles["target_ra"]) & (
        target_tiles["target_ra"] < target_tiles["ra_max"]
    )
    within_dec = (target_tiles["dec_min"] < target_tiles["target_dec"]) & (
        target_tiles["target_dec"] < target_tiles["dec_max"]
    )
    target_tiles["within_tile"] = within_ra & within_dec
    logging.info(
        f'Targets within tile FoV: {target_tiles["within_tile"].sum()} of {len(target_tiles)}'
    )

    # filter to only those tiles
    target_tiles = target_tiles[target_tiles["within_tile"]]
    assert len(target_tiles) > 0, "No targets within FoV of any tiles, likely a bug"
    # simplify/explicit for export
    target_tiles = target_tiles[
        ["tile_index", "id_str", "target_ra", "target_dec", "target_field_of_view"]
    ]
    assert len(target_tiles) > 0

    # target tiles says which tile (index) to use for each target
    return target_tiles


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
    for tile_index in targets_with_tiles["tile_index"].unique():
        try:
            dict_of_locs = download_all_data_at_tile_index(cfg, tile_index)
            logging.info(f"Downloaded: {dict_of_locs}")
        except AssertionError as e:
            logging.critical(f"Error downloading tile data for {tile_index}")
            logging.critical(e)
            raise e

        targets_at_that_index = targets_with_tiles.query(f"tile_index == {tile_index}")

        save_cutouts_for_all_targets_in_that_tile(
            cfg, dict_of_locs, targets_at_that_index
        )


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

    logging.info(f"Downloaded flux+auxillary tiles: {dict_of_locs}")
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
        targets_at_that_index (pd.DataFrame): The subset of targets (sources) within that single tile. Columns ["tile_index" (now only one), "id_str", "target_ra", "target_dec", "target_field_of_view"]
    """

    assert targets_at_that_index["tile_index"].nunique() == 1

    cutout_data = {}
    for band in cfg.bands:
        # this is easier to load once (per band) and then look up each target...
        cutout_data[band] = get_cutout_data_for_band(
            cfg, dict_of_locs[band], targets_at_that_index
        )
        # so each cutout_data[band] is a list of dicts, one per target, like [{'FLUX': flux_cutout, 'MERPSF': psf_cutout, ...}, ...]
    # ...but saving fits we want to iterate over targets first, and get the data across all bands
    for target_n, target in targets_at_that_index.iterrows():
        # this reshapes the data to be a nested dict, with the top level keyed by band, and the inner level keyed by product type (exactly like dict_of_locs)
        # e.g. { VIS: {FLUX: flux_cutout, MERPSF: psf_cutout, ...}, NIR_Y: {...}, ...}
        target_data = { band: cutout_data[band][target_n] for band in cfg.bands }
        save_loc = os.path.join(
            cfg.fits_dir, str(target["tile_index"]), str(target["id_str"]) + ".fits"
        )
        save_multifits_cutout(cfg, target_data, save_loc)


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
        psf_tree = KDTree(psf_table[["x_center", "y_center"]])
        psf_wcs = WCS(psf_header)

    cutout_data = []
    for target_n, target in targets_at_that_index.iterrows():
        logging.info(f"target {target_n} of {len(targets_at_that_index)}")

        cutout_data_for_target = {}

        # cut out the flux data
        target_coord = SkyCoord(
            target["target_ra"], target["target_dec"], frame="icrs", unit="deg"
        )
        flux_cutout = Cutout2D(
            data=flux_data,
            position=target_coord,
            size=target["target_field_of_view"] * u.arcsec,
            wcs=flux_wcs,
            mode="partial",
        )
        cutout_data_for_target["FLUX"] = flux_cutout

        if "MERRMS" in cfg.auxillary_products:
            rms_cutout = Cutout2D(
                data=rms_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=rms_wcs,
                mode="partial",
            )
            cutout_data_for_target["MERRMS"] = rms_cutout

        if "MERBKG" in cfg.auxillary_products:
            bkg_cutout = Cutout2D(
                data=bkg_data,
                position=target_coord,
                size=target["target_field_of_view"] * u.arcsec,
                wcs=bkg_wcs,
                mode="partial",
            )
            cutout_data_for_target["MERBKG"] = bkg_cutout

        if "MERPSF" in cfg.auxillary_products:
            # find pixel coordinates of target in PSF tile
            target_pixels = psf_wcs.world_to_pixel(target_coord)
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

            # ends up off center for some reason?
            # WCS used only to have a convenient header for the output file
            psf_cutout = Cutout2D(
                data=psf_tile,
                position=(closest_psf["x_center"], closest_psf["y_center"]),
                size=stamp_size,
                wcs=psf_wcs,
                mode="partial",
            )
            # could alternatively use the RA/DEC
            # cutout_psf = Cutout2D(data=psf_tile, position=(closest_psf['RA'], closest_psf['Dec']), size=stamp_size*u.pix)
            cutout_data_for_target["MERPSF"] = psf_cutout

        cutout_data.append(cutout_data_for_target)
    return cutout_data


def save_multifits_cutout(cfg: OmegaConf, target_data: dict, save_loc: str):
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

    header_hdu = fits.PrimaryHDU()

    hdu_list = [header_hdu]

    for band in cfg.bands:
        band_data = target_data[band]
        cutout_flux = band_data["FLUX"]
        flux_header = cutout_flux.wcs.to_header()
        flux_header.append(
            ("FILTER", band, "The Euclid filter used for this flux image"),
            end=True,
        )
        flux_hdu = fits.ImageHDU(
            data=cutout_flux.data, name=f"{cutout_flux}_FLUX", header=flux_header
        )
        hdu_list.append(flux_hdu)

        # TODO this is a bit lazy/repetitive, could be refactored

        if "MERPSF" in cfg.auxillary_products:
            cutout_psf = band_data["MERPSF"]
            psf_header = cutout_psf.wcs.to_header()
            psf_header.append(
                (
                    "FILTER",
                    band,
                    "The Euclid filter used for this PSF image",
                ),
                end=True,
            )
            psf_hdu = fits.ImageHDU(
                data=cutout_psf.data, name="MERPSF", header=psf_header
            )
            hdu_list.append(psf_hdu)

        if "MERRMS" in cfg.auxillary_products:
            cutout_rms = band_data["MERRMS"]
            rms_header = cutout_rms.wcs.to_header()
            rms_header.append(
                (
                    "FILTER",
                    band,
                    "The Euclid filter used for this RMS image",
                ),
                end=True,
            )
            rms_hdu = fits.ImageHDU(data=cutout_rms.data, name="MERRMS")
            hdu_list.append(rms_hdu)

        if "MERBKG" in cfg.auxillary_products:
            cutout_bkg = band_data["MERBKG"]
            bkg_header = cutout_bkg.wcs.to_header()
            bkg_header.append(
                (
                    "FILTER",
                    band,
                    "The Euclid filter used for this BKG image",
                ),
                end=True,
            )
            bkg_hdu = fits.ImageHDU(data=cutout_bkg.data, name="MERBKG")
            hdu_list.append(bkg_hdu)

    hdul = fits.HDUList(hdu_list)

    if not os.path.isdir(os.path.dirname(save_loc)):
        os.mkdir(os.path.dirname(save_loc))

    hdul.writeto(save_loc, overwrite=True)


def create_folders(cfg: OmegaConf):
    cfg.download_dir = cfg.base_dir + "/" + cfg.name
    cfg.tile_dir = cfg.download_dir + "/tiles"

    cfg.cutout_dir = cfg.download_dir + "/cutouts"
    cfg.fits_dir = cfg.cutout_dir + "/fits"

    cfg.sanity_dir = cfg.download_dir + "/sanity"

    logging.info(f"Saving to {cfg.download_dir}")
    assert os.path.exists(os.path.dirname(cfg.download_dir))
    for d in [
        cfg.download_dir,
        cfg.tile_dir,
        cfg.cutout_dir,
        cfg.fits_dir,
        cfg.sanity_dir,
    ]:
        if not os.path.exists(d):
            os.makedirs(d)

    return cfg
