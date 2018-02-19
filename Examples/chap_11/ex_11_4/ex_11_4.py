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
yobs = ytrue + sigma_d * np.random.randn(nobs)

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
