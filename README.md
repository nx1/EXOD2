# EXOD
X-ray outburst detector
See Github Projects page for more information
https://github.com/users/nx1/projects/2

# Installation
`git clone https://github.com/nx1/EXOD2`

`cd EXOD`

`pip install -e .`

Then set the 'EXOD' enviroment in your .bashrc to point to this repo i.e.
`export EXOD='/home/{username}/EXOD2'`


# Running
The main script for the pipeline is found in EXOD2/exod/main.py
This will run over all the observations specified in 
`EXOD2/data/observations.txt`
and perform the transient search, the output is then saved in
`/data/results/obsid/`


# Code
Classes
=======

`xmm.Observation(obsid)` : Getting Observation Data and setting directory structure
`xmm.EventList(path)` : Reading and checking storing EventList
`pre_processing.DataLoader()` : Creates the DataCube from the EventList (I agree this is clunky)
`processing.DataCube(EventList, size_arcsec, time_interval)` : Creating and storing the datacube from the EventList
`processing.Detector(DataCube, wcs, sigma)` : Main detection algorithm on data cube.
