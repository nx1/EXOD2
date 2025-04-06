from exod.processing.pipeline import Pipeline

if __name__ == "__main__":
    p = Pipeline(obsid='0724210501',
            size_arcsec=20,
            time_interval=300,
            remove_partial_ccd_frames=True,
            min_energy=0.2,
            max_energy=12.0)
    p.run()
    p.load_results()