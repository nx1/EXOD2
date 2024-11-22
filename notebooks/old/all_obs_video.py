from exod.utils.path import read_observation_ids
import exod.utils.path as path
from exod.xmm.observation import Observation
from exod.processing.data_cube import DataCubeXMM 

obsids = read_observation_ids('../data/obs_ccd_check.txt')
for obsid in obsids:
    obsid = '0116700301'
    obs = Observation(obsid)
    obs.get_files()
    for evt in obs.events_processed_pn:
        print(evt)
        evt.read()
        evt.info
        dc = DataCubeXMM(event_list=evt, size_arcsec=20, time_interval=100)
        dc.video()
