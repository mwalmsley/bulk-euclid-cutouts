# bulk-euclid-cutouts

Pipeline to make 1M+ Euclid cutouts on ESA Datalabs.

Documentation on [this Google doc](https://docs.google.com/document/d/10KrelkVQgckFmqHIqVzZ-22oPIKe-uIE_0laMHSl3Rs/edit?usp=sharing).

There are actually two pipelines, with many utils shared by both:

- `bulk_euclid/mer_catalog_targets` downloads human-friendly JPG cutouts for all pretty (bright or highly-extended) sources in the MER Catalog
- `bulk_euclid/external_targets` downloads FITS cutouts (flux, PSF, RMS, background) for a list of external coordinates

These were developed for [Galaxy Zoo Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_Galaxy_Zoo_help_us_classify_the_shapes_of_galaxies) and strong lensing, respectively.

Help wanted. Let's build a pipeline that works for many projects. Contact Mike Walmsley ([m.walmsley@utoronto.ca](emailto:m.walmsley@utoronto.ca))


## Making back into one pipeline?

The basic process is similar: download a tile, slice cutouts, save as directed.

The details in the first half are different: identify sources from the MER catalog, or sources from an external target list? 
But once we have a list of targets: (id_str, ra, dec, FoV) we should be able to do the same download process.



## Quick Install Instructions on Datalabs

Create a GitHub personal access token and save it somewhere. You will need it to clone this private repo.

Create a Datalab with jl-euclid environment

Open a terminal via Jupyter (NOT within a notebook). Navigate to the folder into which you want to clone this repo

    git clone git@github.com:mwalmsley/bulk-euclid-cutouts.git
    cd bulk-euclid-cutouts
    git checkout external-targets

Install a few missing dependencies (omegaconf, sklearn)

    conda activate euclid-tools
    pip install -e .

Now you can open the notebook (`bulk_euclid/external_targets/notebook_version.ipynb`) and it should run with no further setup. 

You might want to change the configuration options (`cfg_dict`, near the top of the notebook), for example

- `base_dir` (the location where all data is saved from all pipeline runs)
- `name` (the subdirectory within `base_dir` where the data from this pipeline run is saved).

For long (hour plus) downloads, we've noticed Datalabs sometimes silently kills notebooks. Use the terminal instead. See `bulk_euclid/external_targets/run_from_console.py`.

    python /media/user/repos/bulk-euclid-cutouts/bulk_euclid/external_targets/run_from_console.py


## Installing locally

    conda create -p .conda python==3.10

Then pip install just like on Datalabs.

If you get "setup.py not found, toml cannot be installed in editable, you need to upgrade pip to > 21.3:

     pip install --upgrade pip

The Euclid package on Datalabs is not yet available outside Datalabs and so you cannot download any Euclid data, unfortunately.