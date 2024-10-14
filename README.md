# bulk-euclid-cutouts

Pipeline to make 1M+ Euclid cutouts on ESA Datalabs.

Documentation on [this Google doc](https://docs.google.com/document/d/10KrelkVQgckFmqHIqVzZ-22oPIKe-uIE_0laMHSl3Rs/edit?usp=sharing).

Currently used for [Galaxy Zoo Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_Galaxy_Zoo_help_us_classify_the_shapes_of_galaxies).

Help wanted. Let's build a pipeline that works for many projects. Contact Mike Walmsley ([m.walmsley@utoronto.ca](emailto:m.walmsley@utoronto.ca))



Mode taking coordinates and FoV and simply making cutouts/FoV, no Euclid catalog required

Inputs:
RA/Dec, FoV information

Identify tiles covering those coordinates (from tile list, with list of allowed DR, pick the closest tile, within diagonal 15x15 arcminutes of tile center)
Create unique tile subset
For each tile, download that tile, and use cutout2d to slice out the relevant fits/PSF



Mode cross-matching to Euclid catalog


Take a list of sky coordinates
Find the closest matching object

Cone search, sorted by separation, with filter for magnitude
Check the source exists correctly (max separation for reasonable cutout)
Later, set the FoV by source property (external catalog?)


Find the tile covering that object (segmap_id)
Repeat for all objects
Download the tiles
Cut out the object and PSF