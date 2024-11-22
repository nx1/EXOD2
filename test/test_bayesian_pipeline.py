from exod.xmm.observation import Observation
from exod.processing.pipeline import Pipeline, parameter_grid
from exod.processing.bayesian_computations import PrecomputeBayesLimits


class TestBayesianPipeline:
    def test_main(self):
        params = {'obsid'                     : '0722430101',
                  'size_arcsec'               : 20.0,
                  'time_interval'             : 200,
                  'gti_only'                  : False,
                  'remove_partial_ccd_frames' : True,
                  'min_energy'                : 0.2,
                  'max_energy'                : 12.0,
                  'clobber'                   : False,
                  'precomputed_bayes_limit'   : PrecomputeBayesLimits(threshold_sigma=3)}
        pipeline = Pipeline(**params)
        pipeline.run()
        pipeline.load_results()

