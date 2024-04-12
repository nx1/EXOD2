![](data/EXOD_Logo.png)

EPIC XMM X-ray outburst detector (EXOD).
# Installation
```
git clone https://github.com/nx1/EXOD2
cd EXOD
pip install -e .
Then set the 'EXOD' enviroment in your .bashrc to point to this repo i.e.
export EXOD='/home/{username}/EXOD2'
```

# Running
The main script for the pipeline is found in EXOD2/exod/main.py
This will run over all the obsids specified in
`EXOD2/data/observations.txt`
and perform the transient search, the output is then saved in
`/data/results/obsid/`
