#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example 11.2
from 'Parameter Estimation and Inverse Problems', 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
"""

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

from peiplib import custompp   # noqa

from bayes import bayes


# -----------------------------------------------------------------------------

# Load in Shaw problem for m=n=20
infile = np.load('shawexamp.npz')
G = infile['G']
dspiken = infile['dspiken']
theta = infile['theta']

m, n = G.shape

# Target model (i.e mtrue)
spike = np.zeros(n, dtype=np.float)
spike[9] = 1.0

# Set the data covariance matrix
noise = 1.0e-6
CD = noise**2 * np.eye(m)

# --- Compute the MAP solution using an uninformative MVN ---

# Setup the prior
mu_m = 0.5
sigma_m = 0.5
mprior = mu_m * np.ones(n)
CM = sigma_m**2 * np.eye(n)

# Compute the posterior distribution
mmap, covmp = bayes(G, mprior, CM, dspiken, CD)

# Plot MAP solution and the true model
fig0, ax0 = plt.subplots(1, 1)
ax0.step(theta, spike, where='mid', label='Target model')
ax0.step(theta, mmap, where='mid', linestyle='--', label='MAP solution')
ax0.set_xlabel(r'$\theta$ [rad]')
ax0.set_ylabel('Intensity')
ax0.set_xlim(-2, 2)
ax0.set_ylim(-0.5, 1.5)
ax0.legend()
fig0.savefig('c11fmmap.pdf')
plt.close()

# 95% probability intervals
prob95 = 1.96 * np.sqrt(np.diag(covmp))

# Plot MAP solution with error bars
fig1, ax1 = plt.subplots(1, 1)
ax1.step(theta, mmap, where='mid', label='MAP solution')
dummy = ax1.step(
    theta, mmap+prob95, where='mid', linestyle='--',
    label=r'95$\%$ probability interval')
color = plt.get(dummy[0], 'color')
ax1.step(theta, mmap-prob95, where='mid', linestyle='--', color=color)
ax1.set_xlabel(r'$\theta$ [rad]')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-1.0, 1.5)
ax1.legend()
fig1.savefig('c11fmmapb.pdf')
plt.close()
