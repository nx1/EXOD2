#
EXOD
X-ray outburst detector
See Github Projects page for more information
https://github.com/users/nx1/projects/2

# Installation
`git clone https://github.com/nx1/EXOD2`

`cd EXOD`

`pip install .`

Then set the 'EXOD' enviroment in your .bashrc to point to this repo i.e.
`export EXOD='/home/{username}/EXOD2'`

# Downloading Data.
Add your observation IDs to 
`EXOD2/data/observations.txt`

Run the script:
`python EXOD2/exod/pre_processing/create_download_script.py`

Next run the .sh file created:
`cd EXOD2/data`
`chmod +x download_obs.sh`
`./download_obs.sh`

This will download the raw PPS XMM files to `data/data_downloaded`

Next extract the downloaded files using 
`python EXOD2/exod/pre_processing/extract_downloaded_files.py`

# 
