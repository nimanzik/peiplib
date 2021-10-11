from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

from peiplib import custompp   # noqa

# -----------------------------------------------------------------------------

# ----- Generate the data set -----

mtrue = [1.0, -0.5, 1.0, -0.75]
nmod = 4
nobs = 25
x = np.linspace(1., 7., nobs)
ytrue = mtrue[0]*np.exp(mtrue[1]*x) + mtrue[2]*x*np.exp(mtrue[3]*x)

sigma_d = 0.01
yobs = ytrue + sigma_d*np.random.randn(nobs)

# ----- Set the MCMC parameters -----

# Number of skips to reduce auto-correlation of models
skip = 1000

# Burn-in steps
burnin = 10000

# Number of posterior distribution samples
N = 410000

# MVN step size
step = 0.005 * np.ones(4)

# Initialize model at a random point on [-1, 1]
m0 = (np.random.rand(nmod) - 0.5) * 2

# ----- MCMC function calls ------


def log_prior(ms):
    """
    Compute the logarithm of prior distribution.

    For this problem, we apply a uniform prior for the region m1=[0, 2],
    m2=[-1, 0], m3=[0, 2], and m4=[-1, 0].
    """
    if (0 <= ms[0] <= 2) and (-0.99 <= ms[1] <= 0) and (0 <= ms[2] <= 2) and \
            (-0.99 <= ms[3] <= 0):
        return 0.
    else:
        return -np.inf


def forward(ms, xs):
    """
    Forward problem, eq. (11.59).
    """
    return ms[0]*np.exp(ms[1]*xs) + ms[2]*xs*np.exp(ms[3]*xs)


def log_likelihood(yobs, ms, xs, sigma):
    """
    Compute the natural logarithm of likelihood function, eq. (11.61).
    """
    return (-0.5) * np.sum((yobs - forward(ms, xs))**2 / sigma**2)


def generate_random_model(ms, step):
    """
    Generate a random model (i.e. candidate model, c) from the current
    model (i.e. previous model, m^j) using the proposal distribution
    r(x, y).

    For this problem, we use a multivariate normal (MVN) generator with
    standard deviations specified by the ``step`` (see eq. (9.3) in
    Calvetti & Somersalo [2007]).
    """
    if np.isscalar(ms):
        if not np.isscalar(step):
            raise ValueError('argument "step" must be a scalar')

        w = np.random.randn()

    else:
        ms = np.asarray(ms)

        if (ms.ndim > 2) or (ms.ndim == 2 and ms.shape[1] != 1):
            raise ValueError(
                'argument "ms" must be a scalar, 1-D array, or a 2-D vector.')

        nmod = len(ms)
        step = np.asarray(step).reshape(-1,)
        if step.size not in [1, nmod]:
            raise ValueError(
                'argument "step" must be either a scalar, or a 1-D array/'
                '2-D vector of length {}'.format(nmod))

        w = np.random.randn(nmod)

    return ms + step*w


def log_proposal(x, y, step):
    """
    Compute the natural logarithm of proposal distribution, r(x, y).
    """
    return (-0.5) * np.sum((x-y)**2 / step**2)
