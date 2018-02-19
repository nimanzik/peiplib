#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example 11.3
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
"""

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from peiplib import custompp   # noqa
from peiplib.bayesian import bayes, corrmat
from peiplib import plot


km2m = 1000.
m2km = 1. / km2m

skyblue = plot.to01(plot.tango_colors['skyblue2'])
aluminium = plot.to01(plot.tango_colors['aluminium5'])
plt.rc('text', usetex=False)


# -----------------------------------------------------------------------------

vsp = loadmat('vsp.mat')

k = 1

# `dobs` is a vector containing the depths of the different observations.
# `dd` is difference in depth between successive observations.
# `t` contains the noisy observations.
depth = vsp['depth']
dobs = vsp['dobs']
t = vsp['t']

# Number of model parameters (set k equal to an integer >=1 to explore
# the (n >= m) case in this example.
m = dobs.shape[0]
n = k * m
borehole = 1. * km2m
dy = borehole / m
dd = dy / k

sigma_d = vsp['noise'].ravel()[0]   # [s]
sigma_m = 2.0e-5   # [s/m]

# -----------------------------------------------------------------------------

# Generate the true smooth model

vel = (3. + np.sqrt(depth*m2km)) * km2m
strue = 1. / vel

# Set up the system matrix

G = np.zeros((m, n), dtype=np.float)
for i in range(m):
    G[i, 0:(i+1)*k] = dd

# Data covariance matrix

CD = sigma_d**2 * np.identity(m)

# Set up the prior

clen = 5
mprior = np.linspace(strue[0], strue[-1], m)
cmat, cfun = corrmat(m, clen, want_corrfun=True)
CM = sigma_m**2 * cmat
prob95_prior = 1.96 * np.sqrt(np.diag(CM))

# Compute the posterior distribution and the MAP solution

mmap, covmp = bayes(G, mprior, CM, t, CD)
prob95_post = 1.96 * np.sqrt(np.diag(covmp))

# Plot the true slowness model and prior distribution
print(
    'Displaying the true slowness model and a constant-slowness gradient '
    'prior distribution (fig. 1)')

dummy_depth = np.linspace(depth.min(), depth.max(), m)

fig0, ax0 = plt.subplots(1, 1)
ax0.plot(depth, strue/m2km, color=skyblue, lw=2)
ax0.step(dummy_depth, mprior/m2km, where='mid')
ax0.step(
    dummy_depth, (mprior+prob95_prior)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax0.step(
    dummy_depth, (mprior-prob95_prior)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax0.set_xlabel('Depth [m]')
ax0.set_ylabel('True slowness [s/km]')
fig0.savefig('c11f09_vspprior.pdf')
plt.close()

# Plot the first correlation function
print ('Displaying the correlation function for the first MAP model (fig. 2)')

fig1, ax1 = plt.subplots(1, 1)
ax1.step(np.linspace(-500, 500, m), cmat[25, :], where='mid')
ax1.set_xlabel('Lag [m]')
ax1.set_ylabel(r'$a_i$')
ax1.set_xlim(-500, 500)
ax1.set_ylim(0., 1.05)
fig1.savefig('c11f10_vsp1corr.pdf')
plt.close()

# Plot the first MAP solution
print('Displaying the first MAP model (fig. 3)')

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(depth, strue/m2km, color=skyblue, lw=2)
ax2.step(dummy_depth, mmap/m2km, where='mid')
ax2.step(
    dummy_depth, (mmap+prob95_post)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax2.step(
    dummy_depth, (mmap-prob95_post)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax2.set_xlabel('Depth [m]')
ax2.set_ylabel('Slowness [s/km]')
ax2.set_ylim(0.22, 0.34)
fig2.savefig('c11f11_vsp1bayes.pdf')
plt.close()

# ---------------

# Set up a smoother prior

clen = 10
mprior = np.linspace(strue[0], strue[-1], m)
cmat, cfun = corrmat(m, clen, want_corrfun=True)
CM = sigma_m**2 * cmat
prob95_prior = 1.96 * np.sqrt(np.diag(CM))

# Compute the posterior distribution and the MAP solution

mmap, covmp = bayes(G, mprior, CM, t, CD)
prob95_post = 1.96 * np.sqrt(np.diag(covmp))

# Plot the second correlation function
print ('Displaying the correlation function for the second MAP model (fig. 4)')

fig3, ax3 = plt.subplots(1, 1)
ax3.step(np.linspace(-500, 500, m), cmat[25, :], where='mid')
ax3.set_xlabel('Lag [m]')
ax3.set_ylabel(r'$a_i$')
ax3.set_xlim(-500, 500)
ax3.set_ylim(0., 1.05)
fig3.savefig('c11f12_vsp2corr.pdf')
plt.close()

# Plot the second MAP solution
print('Displaying the second MAP model (fig. 5)')

fig4, ax4 = plt.subplots(1, 1)
ax4.plot(depth, strue/m2km, color=skyblue, lw=2)
ax4.step(dummy_depth, mmap/m2km, where='mid')
ax4.step(
    dummy_depth, (mmap+prob95_post)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax4.step(
    dummy_depth, (mmap-prob95_post)/m2km,
    where='mid', linestyle='-.', color=aluminium)
ax4.set_xlabel('Depth [m]')
ax4.set_ylabel('Slowness [s/km]')
ax4.set_ylim(0.22, 0.34)
fig4.savefig('c11f13_vsp2bayes.pdf')
plt.close()
