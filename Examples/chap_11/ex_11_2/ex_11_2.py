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
from scipy.signal import hann

from peiplib import custompp   # noqa

from peiplib.bayesian import bayes, simmvn


# -----------------------------------------------------------------------------

# Utility functions and variables

ylim = (-0.5, 1.5)


def get_fig_ax(xlim=None, ylim=None):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel('Intensity')
    ax.set_xlim(xlim or (-2, 2))
    if ylim:
        ax.set_ylim(ylim)

    return (fig, ax)


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

# ----- Compute the MAP solution using an uninformative MVN -----

# Setup the prior
mu_m = 0.5
sigma_m = 0.5
mprior = mu_m * np.ones(n)
CM = sigma_m**2 * np.eye(n)

# Compute the posterior distribution
mmap, covmp = bayes(G, mprior, CM, dspiken, CD)

# Posterior 95% probability interval
prob95 = 1.96 * np.sqrt(np.diag(covmp))

# Generate a random solution
mmap_sims = simmvn(mmap, covmp)

# --- Produce some plots ---

# Plot MAP solution and the true model
print(
    'Displaying the true model and a MAP solution based on an '
    'uninformative prior (fig. 1)')

fig0, ax0 = get_fig_ax(ylim=ylim)
ax0.step(theta, spike, where='mid', label='Target model')
ax0.step(theta, mmap, where='mid', linestyle='--', label='MAP solution')
ax0.legend(fontsize=12)
fig0.savefig('c11f03_mmap.pdf')
plt.close()

# Plot MAP solution with error bars
print('Displaying the MAP solution with 95% confidence interval (fig. 2)')

fig1, ax1 = get_fig_ax(ylim=(-1.0, 1.5))
ax1.step(theta, mmap, where='mid', label='MAP solution')
dummy = ax1.step(theta, mmap+prob95, where='mid', linestyle='--')
color = plt.get(dummy[0], 'color')
ax1.step(
    theta, mmap-prob95, where='mid', linestyle='--', color=color,
    label=r'95$\%$ probability interval')
ax1.legend(fontsize=12)
fig1.savefig('c11f04_mmapb.pdf')
plt.close()

# Plot the random solution
print('Displaying randomly selected realization (fig. 3)')

fig2, ax2 = get_fig_ax()
ax2.step(theta, mmap_sims, where='mid')
fig2.savefig('c11f05_mmapsims.pdf')
plt.close()


# ----- Compute the MAP solution using a more restrictive prior -----

# Setup a more restrictive prior
mprior = np.zeros(n)
mprior[4:-4] = hann(12, sym=True)

CM = 0.25**2 * np.diag((hann(n+2, sym=True)[1:-1])**2)

# Prior 95% probability interval
prob95_prior = 1.96 * np.sqrt(np.diag(CM))

# Compute the posterior distribution
mmap, covmp = bayes(G, mprior, CM, dspiken, CD)

# Posterior 95% probability interval
prob95_post = 1.96 * np.sqrt(np.diag(covmp))

# Generate a random solution
mmap_sims = simmvn(mmap, covmp)


# --- Produce some plots ---

# Plot prior distribution
print(
    'Displaying the prior distribution and a 95% probability '
    'interval around it (fig. 4)')

fig3, ax3 = get_fig_ax(ylim=ylim)
ax3.step(theta, mprior, where='mid', label='Prior distribution')
dummy = ax3.step(theta, mprior+prob95_prior, where='mid', linestyle='--')
color = plt.get(dummy[0], 'color')
ax3.step(
    theta, mprior-prob95_prior, where='mid', color=color, linestyle='--',
    label=r'95$\%$ probability interval')
ax3.legend(fontsize=12)
fig3.savefig('c11f06_priorr.pdf')
plt.close()

# Plot MAP solution and the true model
print(
    'Displaying the true model and a MAP solution based on prior that'
    'expects a central spike (fig. 5)')

fig4, ax4 = get_fig_ax(ylim=ylim)
ax4.step(theta, spike, where='mid', label='Target model')
ax4.step(theta, mmap, where='mid', linestyle='--', label='MAP solution')
ax4.legend(fontsize=12)
fig4.savefig('c11f07_mmapr.pdf')
plt.close()

# Plot MAP solution with error bars
print('Displaying the MAP solution with 95% confidence interval (fig. 6)')

fig5, ax5 = get_fig_ax(ylim=(-1.0, 1.5))
ax5.step(theta, mmap, where='mid', label='MAP solution')
dummy = ax5.step(theta, mmap+prob95_post, where='mid', linestyle='--')
color = plt.get(dummy[0], 'color')
ax5.step(
    theta, mmap-prob95_post, where='mid', linestyle='--', color=color,
    label=r'95$\%$ probability interval')
ax5.legend(fontsize=12)
fig5.savefig('c11f08_mmapbr.pdf')
plt.close()

# Plot the random solution
print('Displaying randomly selected realization (fig. 7)')

fig6, ax6 = get_fig_ax()
ax6.step(theta, mmap_sims, where='mid')
fig6.savefig('c11fXX_mmapsimsr.pdf')
plt.close()
