import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.special import gammaincc, gammainc
from scipy.stats import poisson
from tqdm import tqdm

from exod.utils import path as path
from exod.utils.logger import logger
from exod.utils.path import data_plots


def B_peak(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the data_cube (mu) and observed (n) counts."""
    return gammaincc(n + 1, mu) / poisson.pmf(n, mu)


def B_eclipse(n, mu):
    """Computes the Bayes factors of the presence of an eclipse, given the data_cube (mu) and observed (n) counts"""
    return gammainc(n + 1, mu) / poisson.pmf(n, mu)


def B_peak_log(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the data_cube (mu) and observed (n) counts."""
    return np.log10(gammaincc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def B_eclipse_log(n, mu):
    """Computes the Bayes factors of the presence of an eclipse, given the data_cube (mu) and observed (n) counts"""
    return np.log10(gammainc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def n_peak_large_mu(mu, sigma):
    """Calculate the observed (n) value required for a peak at a specific expectation (mu) and significance (sigma) in the Gaussian regime."""
    return np.ceil((2*mu+sigma**2+np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def n_eclipse_large_mu(mu, sigma):
    """Calculate the observed (n) value required for an eclipse at a specific expecation (mu) and significance (sigma) in the Gaussian regime."""
    return np.floor((2*mu+sigma**2-np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def precompute_bayes_limits(threshold_sigma):
    """
    Compute the minimum and maximum number of observed
    counts (n) required for an eclipse or peak for a given
    confidence threshold (threshold_sigma) and expectation (mu).

    For counts > 1000, we use a Gaussian approximation.
    sigma = (N-mu) / (N+mu)^0.5

    Solving for N gives:
    N = 2mu + sigma^2 + (8 mu sigma^2 + sigma^4)^0.5
    N = 2mu + sigma^2 - (8 mu sigma^2 + sigma^4)^0.5

    The resulting table looks like:
        i          mu  n_peak  n_eclipse
    46000  158.556483   219.0      106.0
    46001  158.629519   219.0      107.0
    46002  158.702589   219.0      107.0
    46003  158.775693   219.0      107.0


    Each data cell in the observed and data_cube cube
    is then compared to the values pre-calculated here
    to determine if is it a peak or an eclipse.

    An example for 1 frame is given here.

     (n_cube)    (mu_cube)       is_3_sig_peak?
      0 2 1    0.03 1.98 0.87       F F F
      0 5 0    1.01 1.10 1.00       F T F
      3 2 1    2.30 2.14 0.98       F F F

    The evaluation of each data cell of the cube against
    a threshold is made faster by precomputing the counts here.
    """
    B_peak_threshold, B_eclipse_threshold = get_bayes_thresholds(threshold_sigma=threshold_sigma)

    range_mu       = np.geomspace(start=1e-7, stop=1e3, num=50000)
    range_mu_large = np.geomspace(start=1e3, stop=1e6, num=1000)   # Above 1000

    tab_npeak, tab_neclipse = [], []
    for mu in tqdm(range_mu):
        # Find the smallest observed (n) value for a peak
        range_n_peak = np.arange(max(10 * int(mu), 100))
        B_factors = B_peak_log(n=range_n_peak, mu=mu)
        n_peak_min = range_n_peak[B_factors > B_peak_threshold][0]
        tab_npeak.append(n_peak_min)

        # Get the largest observed (n) value for an eclipse
        range_n_eclipse = np.arange(2 * int(mu) + 1)
        B_factors = B_eclipse_log(n=range_n_eclipse, mu=mu)
        n_eclipse_max = range_n_eclipse[B_factors < B_eclipse_threshold][0]
        tab_neclipse.append(n_eclipse_max)

    tab_npeak    += list(n_peak_large_mu(range_mu_large, threshold_sigma))
    tab_neclipse += list(n_eclipse_large_mu(range_mu_large, threshold_sigma))

    range_mu = np.concatenate((range_mu, range_mu_large))

    data = np.array([range_mu, tab_npeak, tab_neclipse])
    savepath = path.utils / f'bayesfactorlimits_{threshold_sigma}.txt'
    logger.info(f'Saving to {savepath}')
    np.savetxt(savepath, data)


def plot_bayes_limits():
    pbl_3 = PrecomputeBayesLimits(3)
    pbl_5 = PrecomputeBayesLimits(5)

    range_mu  = pbl_3.range_mu

    tab_npeak_3sig = pbl_3.n_peak_threshold(range_mu)
    tab_neclipse_3sig = pbl_3.n_eclipse_threshold(range_mu)

    tab_npeak = pbl_5.n_peak_threshold(range_mu)
    tab_neclipse = pbl_5.n_eclipse_threshold(range_mu)


    plt.figure(figsize=(4, 4))

    plt.plot(range_mu, tab_npeak, ls=':', c='k', label=fr'5$\sigma$', lw=1.0)
    plt.plot(range_mu, tab_neclipse, ls=':', c='k', lw=1.0)

    plt.plot(range_mu, tab_npeak_3sig, ls='--', c='k', label=fr'3$\sigma$', lw=1.0)
    plt.plot(range_mu, tab_neclipse_3sig, ls='--', c='k', lw=1.0)

    plt.fill_between(range_mu, tab_npeak, 1e6, alpha=0.5, color='steelblue', label='Detection Region')
    plt.fill_between(range_mu, tab_npeak_3sig, 1e6, alpha=0.3, color='steelblue')
    plt.fill_between(range_mu, 0, tab_neclipse, alpha=0.5, color='steelblue')
    plt.fill_between(range_mu, 0, tab_neclipse_3sig, alpha=0.3, color='steelblue')


    plt.plot(range_mu, range_mu, label=r'$N=\mu$', color='black')
    #plt.fill_between(range_mu, range_mu-5*np.sqrt(range_mu), range_mu+5*np.sqrt(range_mu), alpha=0.3, label=fr'Naive $5 \sigma$', color='grey')
    #plt.fill_between(range_mu, range_mu-3*np.sqrt(range_mu), range_mu+3*np.sqrt(range_mu), alpha=0.5, label=fr'Naive $3 \sigma$', color='grey')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(r'$B_{peak} = \frac{Q(N+1, \mu)}{e^{-\mu} \mu^{N} / N!} \ \  B_{eclipse} = \frac{P(N+1, \mu)}{e^{-\mu} \mu^{N} / N!}$')
    plt.xlabel(fr'Expected Counts $\mu$')
    plt.ylabel(fr'Observed Counts $N$')
    plt.xlim(min(range_mu), max(range_mu))
    plt.ylim(min(range_mu), max(range_mu))
    plt.text(0.05, 25, s='Significant Peaks', )
    plt.text(35, 0.1, s='Significant\nEclipses')
    #plt.yticks([1, 10, 100, 300], labels=[1, 10, 100, 300])
    plt.yticks([0.01, 0.1, 1, 10, 100, 300], labels=[0.01, 0.1, 1, 10, 100, 300])
    plt.xticks([0.1, 1, 10, 100, 300], labels=[ 0.1, 1, 10, 100, 300])
    plt.xlim(0.01, 300)
    plt.ylim(0.01, 300)
    plt.legend(loc='upper left', fontsize=10, ncol=2, columnspacing=0.8)
    #plt.tight_layout()
    plt.savefig(data_plots / f'bayesfactorlimits_3_5.pdf')
    plt.savefig(data_plots / f'bayesfactorlimits_3_5.png')
    plt.show()


def get_cube_masks_peak_and_eclipse(cube_n, cube_mu, threshold_sigma):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse."""
    range_mu, n_peak_threshold, n_eclipse_threshold = load_precomputed_bayes_limits(threshold_sigma=threshold_sigma)
    cube_mu = np.where(cube_mu > range_mu[0], cube_mu, np.nan) # Remove small expectation values outside of interpolation range
    cube_mask_peaks   = cube_n > n_peak_threshold(cube_mu)
    cube_mask_eclipse = cube_n < n_eclipse_threshold(cube_mu)
    return cube_mask_peaks, cube_mask_eclipse

def precompute_bayes_1000():
    """Precomputes the Bayes factor at mu=1000 for a bunch of values of N. Will be interpolated to estimate the sigma"""
    range_N    = np.arange(10000)
    B_peaks    = B_peak_log(n=range_N, mu=1000)
    B_eclipses = B_eclipse_log(n=range_N, mu=1000)
    data = np.array([range_N, B_peaks, B_eclipses])
    savepath = path.utils / f'bayesfactor_mu1000.txt'
    logger.info(f'Saving to {savepath}')
    np.savetxt(savepath, data)


def load_precomputed_bayes1000():
    """Loads & interpolates the precomputed values of Bayes factors at mu=1000"""
    data              = np.loadtxt(path.utils / f'bayesfactor_mu1000.txt')
    range_N           = data[0]
    B_values_peaks    = interp1d(range_N, data[1])
    B_values_eclipses = interp1d(range_N, data[2])
    return range_N, B_values_peaks, B_values_eclipses


def load_precomputed_bayes_limits(threshold_sigma):
    """Loads the precomputed Bayes factor limit numbers, for a chosen threshold."""
    data = np.loadtxt(path.utils / f'bayesfactorlimits_{threshold_sigma}.txt')
    range_mu            = data[0]
    n_peak_threshold    = interp1d(range_mu, data[1])
    n_eclipse_threshold = interp1d(range_mu, data[2])
    return range_mu, n_peak_threshold, n_eclipse_threshold


def get_bayes_thresholds(threshold_sigma):
    """
    The thresholds for B_peak and B_eclipse are calculated here for 3 and 5 sigma.

    This is sort of a hack, and is done by finding the value of B(n,mu) that is equal to
    a given significance (sigma) under the Gaussian assumption at mu=1000.

    For example, we want a significance level of sigma = 3.
    We first find what observed (n) value we need to get a 3 sigma peak Gaussian assumption.
        n_peak_large_mu(mu=1000, sigma=3) = 1139

    Next, we find the value of B_peak that an observed (n) value of 1139 would give us.
        B_peak(n=1139, mu=1000) = 872908  (5.94 in log10)

    We treat this value of B as being "Equivalent" to a 3 sigma detection and subsequently can specify:
        B > B_threshold_peak for a peak detection
        B < B_eclipse_threshold for an eclipse detection
    """
    B_peak_threshold    = B_peak_log(n=n_peak_large_mu(mu=1000, sigma=threshold_sigma), mu=1000)
    B_eclipse_threshold = B_eclipse_log(n=n_eclipse_large_mu(mu=1000, sigma=threshold_sigma), mu=1000)
    return B_peak_threshold, B_eclipse_threshold


def sigma_equivalent(n, mu):
    """
    Find the equivalent sigma for a given observed (n) and expectation (mu).

    For large counts (mu=1000), the required n to obtain a given sigma can be calculated assuming Gaussian statistics.
    This can be done using the functions: n_peak_large_mu(mu, sigma) and n_eclipse_large_mu(mu, sigma)

    For small counts, we cannot use these functions, as we are in the Poisson regime, but we can calculate the
    Bayes factors for peaks and eclipses, using B_peak_log(n, mu) and B_eclipse_log(n, mu).

    We can however for a given value of B, find out what the equivalent value of sigma would be.
    To do this we need to find the sigma that gives:
        B_peak(n, mu=1000) = B_peak(n=n_peak_large_mu(mu=1000, sigma), mu=1000)

    This is done by finding the root of the function:
        B_peak(n, mu=1000) - B_peak(n=n_peak_large_mu(mu=1000, sigma), mu=1000) = 0
    """
    if n > mu:  # Peak
        b = B_peak_log(n, mu)
        function_to_invert = lambda sigma: b - B_peak_log(n_peak_large_mu(mu=1000, sigma=sigma), mu=1000)
    else:  # Eclipse
        b = B_eclipse_log(n, mu)
        function_to_invert = lambda sigma: b - B_eclipse_log(n_eclipse_large_mu(mu=1000, sigma=sigma), mu=1000)

    if function_to_invert(10) > 0:
        return 10
    elif function_to_invert(1) < 0:
        return 0
    else:
        return root_scalar(function_to_invert, bracket=(1, 10)).root


def sigma_equivalent_B_peak(B_peak):
    """
    Find the equivalent sigma for a given Bayes factor for a peak. B_Peak must be in log!

    Range: 1.61 < B_peak < 48.976
           0.00 < sigma  < 9.98
    """
    if B_peak< 1.61:
        return 0
    elif B_peak > 48.976:
        return 10
    f = lambda sigma: B_peak - B_peak_log(n_peak_large_mu(mu=1000, sigma=sigma), mu=1000)
    return root_scalar(f, bracket=(0, 10)).root


def sigma_equivalent_B_eclipse(B_eclipse):
    """
    Find the equivalent sigma for a given Bayes factor for an eclipse. B_eclipse must be in log!

    Range: 1.6  < B_eclipse < 42
           0.00 < sigma     < 9.94
    """
    if B_eclipse < 1.6:
        return 0
    elif B_eclipse > 42:
        return 10
    f = lambda sigma: B_eclipse - B_eclipse_log(n_eclipse_large_mu(mu=1000, sigma=sigma), mu=1000)
    return root_scalar(f, bracket=(0, 10)).root


class PrecomputeBayesLimits:
    _instances = {}

    def __new__(cls, threshold_sigma):
        """
        Creates or retrieves an instance of PrecomputeBayesLimits with a specific threshold.

        This method implements a multiton pattern, where only one instance of
        PrecomputeBayesLimits exists per unique value of `threshold_sigma`.

        When a new threshold value is requested, a new instance is created and stored in
        the `_instances` dictionary, keyed by `threshold_sigma`. If an instance with the same `threshold_sigma`
        already exists, it is returned instead of creating a new one. https://en.wikipedia.org/wiki/Multiton_pattern

        Args:
            threshold_sigma (int): The significance threshold for the instance.

        Returns:
            PrecomputeBayesLimits: The existing or newly created instance with  the specified `threshold_sigma`.
        """
        if threshold_sigma not in cls._instances:
            instance = super(PrecomputeBayesLimits, cls).__new__(cls)
            instance.threshold_sigma = threshold_sigma
            logger.warning(f'Creating new precomputed PrecomputeBayesLimits() instance threshold_sigma={threshold_sigma}.')
            cls._instances[threshold_sigma] = instance
        return cls._instances[threshold_sigma]

    def __init__(self, threshold_sigma):
        self.threshold_sigma = threshold_sigma
        self.get_savepath()
        self.range_mu = None
        self.n_peak_threshold = None
        self.n_eclipse_threshold = None
        self.is_loaded = False
        self.load()

    def __repr__(self):
        return f'PrecomputeBayesLimits(threshold_sigma={self.threshold_sigma})'

    def get_savepath(self):
        self.savepath = path.utils / f'bayesfactorlimits_{self.threshold_sigma}.txt'
        if not self.savepath.exists():
            logger.info(f'{self.savepath} does not exist. Precomputing Bayes Factors...')
            precompute_bayes_limits(threshold_sigma=self.threshold_sigma)
        return self.savepath

    def load(self):
        if self.is_loaded:
            return None
        range_mu, n_peak_threshold, n_eclipse_threshold = load_precomputed_bayes_limits(threshold_sigma=self.threshold_sigma)
        self.range_mu = range_mu
        self.n_peak_threshold = n_peak_threshold
        self.n_eclipse_threshold = n_eclipse_threshold
        self.is_loaded = True

    def get_cube_masks_peak_and_eclipse(self, cube_n, cube_mu):
        cube_mu = np.where(cube_mu > self.range_mu[0], cube_mu, np.nan)  # Remove small expectation values outside of interpolation range
        cube_mask_peaks   = cube_n > self.n_peak_threshold(cube_mu)
        cube_mask_eclipse = cube_n < self.n_eclipse_threshold(cube_mu)
        return cube_mask_peaks, cube_mask_eclipse


if __name__ == "__main__":
    from exod.utils.plotting import use_scienceplots
    use_scienceplots()
    pre = PrecomputeBayesLimits(threshold_sigma=3)
    plot_bayes_limits()

