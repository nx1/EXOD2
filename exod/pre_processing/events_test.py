from exod.xmm.observation import Observation
from exod.utils.path import read_observation_ids

#dl = DataLoader(time_interval=50,
#                size_arcsec=15,
#                gti_only=True,
#                min_energy=0.2,
#                max_energy=12.0,
#                gti_threshold=0.5)
#dl.make_data_cube()



if __name__ == "__main__":
    import pandas as pd
    obsids = read_observation_ids('../../data/observations.txt')

    all_info = []
    for obsid in obsids:
        print(obsid)
        obs = Observation(obsid)
        obs.get_files()

        for evt in obs.events_processed:
            evt.read()
            evt_info = evt.info
            all_info.append(evt_info)

    df_info = pd.DataFrame(all_info)
    print(df_info)


