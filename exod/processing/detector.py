from exod.processing import variability

class Detector:
    def __init__(self, data_loader, wcs):
        self.data_loader = data_loader
        self.data_cube = self.data_loader.data_cube
        self.wcs = wcs


    def __repr__(self):
        return f'Detector({self.data_cube})'

    def run(self):
        self.var_img = variability.calc_var_img(self.data_cube.data)
        self.df_regions = variability.extract_var_regions(self.var_img)
        self.df_regions = variability.get_regions_sky_position(self.df_regions)
        self.df_regions = variability.filter_df_regions(self.df_regions)

        self.save_df_regions()

    def save_df_regions(self):
        pass


    @property
    def info(self):
        info = {'data_loader' : repr(self.data_loader),
                'data_cube'   : repr(self.data_cube),
                'df_regions'  : self.df_regions.info()}
        for k, v in info.items():
            print(f'{k:<11} : {v}')
        return info


if __name__ == "__main__":
    from exod.xmm.observation import Observation
    from exod.processing.data_cube import DataCube
    from exod.pre_processing.data_loader import DataLoader

    obs = Observation('0911791101')
    obs.get_files()
    evt = obs.events_processed[0]
    evt.read()
    img = obs.images[0]
    img.read(wcs_only=True)
    dl = DataLoader(evt)
    dl.run()
    data_cube = dl.data_cube
    detector = Detector(data_loader=dl, wcs=img.wcs)
    print(detector.wcs)
    detector.run()
    detector_info = detector.info

    print(detector)