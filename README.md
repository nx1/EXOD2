# EXOD
X-ray outburst detector
See Github Projects page for more information
https://github.com/users/nx1/projects/2

# Installation
`git clone https://github.com/nx1/EXOD2`

`cd EXOD`

`pip -e install .`

Then set the 'EXOD' enviroment in your .bashrc to point to this repo i.e.
`export EXOD='/home/{username}/EXOD2'`


# Running
The main script for the pipeline is found in EXOD2/exod/main.py
This will run over all the observations specified in 
`EXOD2/data/observations.txt`
and perform the transient search, the output is then saved in
`/data/results/obsid/`

# Downloading Data.
Add your observation IDs to 
`EXOD2/data/observations.txt`

Run the script:
`python EXOD2/exod/pre_processing/create_download_script.py`

Next run the .sh file created:
`cd EXOD2/data`
`chmod +x download_obs.sh`
`./download_obs.sh`

This will download the raw XMM event files to `data/raw`, and will also extract
any tar.gz files if there are multiple poitings in a single observation.
