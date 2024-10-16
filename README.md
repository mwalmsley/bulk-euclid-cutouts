# bulk-euclid-cutouts

Pipeline to make 1M+ Euclid cutouts on ESA Datalabs.

Documentation on [this Google doc](https://docs.google.com/document/d/10KrelkVQgckFmqHIqVzZ-22oPIKe-uIE_0laMHSl3Rs/edit?usp=sharing).

There are actually two pipelines, with many utils shared by both:

- `bulk_euclid/mer_catalog_targets` downloads human-friendly JPG cutouts for all pretty (bright or highly-extended) sources in the MER Catalog
- `bulk_euclid/external_targets` downloads FITS cutouts (flux, PSF, RMS, background) for a list of external coordinates

These were developed for [Galaxy Zoo Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_Galaxy_Zoo_help_us_classify_the_shapes_of_galaxies) and strong lensing, respectively.

Help wanted. Let's build a pipeline that works for many projects. Contact Mike Walmsley ([m.walmsley@utoronto.ca](emailto:m.walmsley@utoronto.ca))
