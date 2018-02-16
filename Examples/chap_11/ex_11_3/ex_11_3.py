#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example 4.4-5
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
"""

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from peiplib import custompp   # noqa


km2m = 1000.
m2km = 1. / km2m


# -----------------------------------------------------------------------------

vsp = loadmat('vsp.mat')

k = 1

depth = vsp['depth']
noise = vsp['noise'].ravel()[0]
dobs = vsp['dobs']
t = vsp['t']

# Generate the true smooth model

vel = (3. + np.sqrt(depth*m2km)) * km2m
strue = 1. / vel

# Set up the system matrix

m = dobs.shape[0]
n = k * m
borehole = 1. * km2m
dy = borehole / m
dd = dy / k
G = np.zeros((m, n), dtype=np.float)
for i in range(m):
    G[i, 0:(i+1)*k] == dd

# Data covariance matrix

CD = noise**2 * np.identity(m)

# Plot the true slowness model
print('Displaying the true slowness model (fig. 1)')

fig0, ax0 = plt.subplots(1, 1)
ax0.plot(depth, strue*1000., 'k', alpha=0.5, lw=2)
ax0.set_xlabel('Depth [m]')
ax0.set_ylabel('True slowness [s/km]')
fig0.savefig('c11f09_vspprior.pdf')
plt.close()
