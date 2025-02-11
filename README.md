<p align="center">
  <img src="docs/images/EXOD_Logo.png" alt="EXOD Logo">
</p>

# EXOD (EPIC XMM-Newton Outburst Detector)

EXOD is a pipeline and detection algorithm designed for the detection of rapid
transients in archival XMM-Newton data.

See documentation at [exod2.readthedocs.io](https://exod2.readthedocs.io/)

[![Documentation Status](https://readthedocs.org/projects/exod2/badge/?version=latest)](https://exod2.readthedocs.io/en/latest/?badge=latest)

# Possible Extensions
```
[ ] Pipeline: Make the region identifier simply "('0724210501_0_5_2.0_12.0', 4)" --> '0724210501_0_5_2.0_12.0_4'

[ ] Benchmarking: Redo the detection fraction plot using skoptimize.
[ ] Benchmarking: Estimate the positional error by crossmatching a subset of confirmed sources with a catalog with the smallest errors, eg GAIA.
[ ] Benchmarking: Once the error on the positional crossmatch we can also calculate the statistics on the 5 sigma sources seperately
[ ] Benchmarking: Half-bin shift of the binning grid to elimitate spurious detections / find diluted signals over two bins.
[ ] Benchmarking: When evaluating the false-positives the distribution of seperation may be fit with two models, however this requires you have the number of sources withing a crossmatch circle of r.

[ ] Extensions: Crossmatch with the stacked xmm catalogue as this has 20% more detections that will be around the limit.
[ ] Extensions: Clustering once at the start vs clustering for each subset can yield different numbers, may be worth looking into this.
[ ] Extensions: Come up with a way of having consistent cluster identification between different clustering runs.
[ ] Extensions: LC Feature that identifies high start or end of lc.
[ ] Extensions: Allow new results to be added to existing results.
[ ] Extensions: Calculate the flux, hardness ratio for each source --> convert_count_to_flux()
[ ] Extensions: Run exod at 600s on full band 0.2-2.0
[ ] Extensions: Run on the iron line band (6-8keV)
[ ] Extensions: K-means clustering on the actual lightcurve data, not extracted features, (see tslearn). Otherwise, maybe look at some of the features that are extracted from tsfresh.
[ ] Extensions: Machine Learning to identify instrumental forms of variability
[ ] Extensions: Template Creation: Check to see if it is worth blurring the image twice, and only do so if we need to.
[ ] Extensions: Have a look at https://github.com/samuelperezdi/umlcaxs see if you can get it running. https://arxiv.org/pdf/2401.12203
[ ] Extensions: Could run on ASCA data https://heasarc.gsfc.nasa.gov/docs/asca/ahp_archive.html

[ ] Paper: Add peak flux to interesting source list.
[ ] Paper: Extract QPE spectrum in 0.3-1.0 and 1-2 then plot the HID.
[ ] Paper: Fit a blackbody to the IR points for the tornado burst.

[ ] Discussion: Check Miniutti paper, try and make the lightcurve that is binned by energy for the "QPE" make a plot that shows the frames before and after a burst to prove that its real.
[ ] Discussion: Add image from ESO of the QPE galaxy https://archive.eso.org/scienceportal/home 23:54:40.68 -37:30:21.30
[ ] Discussion: about the different types of objects that are picked out by EXOD, and if they are new.
[ ] Discussion: One way we can compare the distribution of sources from each catalogue is by using a chi-squared homogeneity test.
```
