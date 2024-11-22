from exod.xmm.observation import Observation


def test_observation():
    obs = Observation(obsid='0722430101')
    obs.make_dirs()
    obs.download_events()
    obs.create_images()
    obs.filter_events()
    obs.get_files()
    obs.get_events_overlapping_subsets()
    obs.info
        
