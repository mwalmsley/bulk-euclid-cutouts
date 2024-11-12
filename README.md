# bulk-euclid-cutouts

Pipeline to make 1M+ Euclid cutouts on ESA Datalabs.

Background documentation on [this Google doc](https://docs.google.com/document/d/10KrelkVQgckFmqHIqVzZ-22oPIKe-uIE_0laMHSl3Rs/edit?usp=sharing). It's a little old now (pre-Q1) but adds some background. For an up-to-date quickstart, keep reading.

There are actually two pipelines, with many utils shared by both:

- `bulk_euclid/mer_catalog_targets` downloads human-friendly JPG cutouts for all pretty (bright or highly-extended) sources in the MER Catalog
- `bulk_euclid/external_targets` downloads FITS cutouts (flux, PSF, RMS, background) for a list of external coordinates

These were developed for [Galaxy Zoo Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_Galaxy_Zoo_help_us_classify_the_shapes_of_galaxies) and strong lensing, respectively.

Help wanted. Let's build a pipeline that works for many projects. Contact Mike Walmsley ([m.walmsley@utoronto.ca](emailto:m.walmsley@utoronto.ca))



## Quick Install/Run Instructions on Datalabs

Create a GitHub personal access token and save it somewhere. You will need it to clone this private repo.

Create a Datalab with the Euclid environment

Open a terminal via Jupyter (NOT within a notebook). Navigate to the folder into which you want to clone this repo

    git clone git@github.com:mwalmsley/bulk-euclid-cutouts.git
    cd bulk-euclid-cutouts
    git checkout external-targets

Install a few missing dependencies (omegaconf, sklearn)

    conda activate euclid-tools
    pip install -e .

Now you can open the notebook (`bulk_euclid/external_targets/tutorial_notebook.ipynb`) and it should run with no further setup.

You might want to change the configuration options (`cfg_dict`, near the top of the notebook), for example

- `base_dir` (the location where all data is saved from all pipeline runs)
- `name` (the subdirectory within `base_dir` where the data from this pipeline run is saved).

The locations to save files is set by these two options. Files will be saved at:

    {base_dir}/{name}/cutouts/jpg/{jpg_format}/{tile}/foo.jpg
    {base_dir}/{name}/cutouts/fits//{tile}/foo.jpg

To change the paths, you only need to set those two options. For example, here is the paths used for downloading Space Warps strong lens candidates (please don't copy/paste these):

    base_dir: /media/home/team_workspaces/Euclid-Consortium/data/strong_lensing/external_targets_pipeline  # general folder for pipeline runs
    name: external_targets_master_list_q1  # folder for a specific pipeline run

For long (hour plus) downloads, we've noticed Datalabs sometimes silently kills notebooks. Use the terminal instead. See `bulk_euclid/external_targets/run_from_console.py`.

    python /media/user/repos/bulk-euclid-cutouts/bulk_euclid/external_targets/run_from_console.py


## Installing locally

    conda create -p .conda python==3.10

Then pip install just like on Datalabs.

If you get "setup.py not found, toml cannot be installed in editable, you need to upgrade pip to > 21.3:

     pip install --upgrade pip

The Euclid package on Datalabs is not yet available outside Datalabs and so you cannot download any Euclid data, unfortunately.
