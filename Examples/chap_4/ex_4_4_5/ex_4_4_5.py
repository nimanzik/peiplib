#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Example 4.4-5
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy.io import loadmat

import custompp
from gsvd import gsvd
from reg import get_reg_mat, lcurve_corner, lcurve_tikh_gsvd


km2m = 1000.0
m2km = 0.001

# -----------------------------------------------------------------------------

vsp = loadmat('vsp.mat')

k = 1

# Number of model parameters (set k equal to an integer >=1 to explore the
# n>m case in this example.
borehole = 1 * km2m
M = 50

dy = borehole / M
dd = dy / k
N = k * M

# --- Generate the true smooth model ---
depth = np.arange(0, borehole+1, 1)
vel = (3 + np.sqrt(depth*m2km)) * km2m
strue = 1.0 / vel


# --- Create the noisy observed data ---
# `dobs` is a vector containing the depths of the different observations.
# `dd` is difference in depth between successive observations.
# `t` contains the noisy observations.
noise = 2.0e-4
dobs = vsp['dobs']
t = vsp['t']


# --- System matrix G ---
G = np.zeros((M, N), dtype=np.float)
for i in range(M):
    G[i, 0:(i+1)*k] = dd


# --- Example 4.4 ---

# Plot the true model
fig0, ax0 = plt.subplots(1, 1)
ax0.plot(depth, strue*1000.0)
ax0.set_xlabel('Depth [m]')
ax0.set_ylabel('True slowness [s/km]')
fig0.savefig('c4fvspmod.pdf')
plt.close()


# --- Compute the least-squares solution of G ---
if M == N:
    m_ls = la.solve(G, t)
else:
    m_ls = la.lstsq(G, t)[0]

# Plot the least-squares solution and confidence intervals
covm_ls = (noise**2) * la.inv(np.dot(G.T, G))
conf95 = 1.96 * np.sqrt(np.diag(covm_ls))
conf95 = conf95.reshape(N, 1)

dz = dobs[1] - dobs[0]
dobs2 = dobs + dz/2

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(dobs2, m_ls*1000, '-', drawstyle='steps')
dum = ax1.plot(dobs2, (m_ls+conf95)*1000, '--', drawstyle='steps')
color = plt.get(dum[0], 'color')
ax1.plot(dobs2, (m_ls-conf95)*1000, '--', color=color, drawstyle='steps')
ax1.set_xlabel('Depth [m]')
ax1.set_ylabel('Slowness [s/km]')
fig1.savefig('c4fmL2.pdf')
plt.close()


# --- Apply first-order Tikhonov regularization ---

L1 = get_reg_mat(N, 1, full=True)
U1, V1, X1, LAM1, MU1 = gsvd(G, L1)


# Apply the L curve criteria to the first-order regularization problem

rho1, eta1, reg_params1 = lcurve_tikh_gsvd(U1, X1, LAM1, MU1, t, G, L1, 1200)
alpha_tikh1, icorner1, _ =  lcurve_corner(rho1, eta1, reg_params1)
rho_corner1 = rho1[icorner1]
eta_corner1 = eta1[icorner1]

print('1st-order regularization parameter is:', alpha_tikh1)


# Get the desired model.
dum1 = np.dot(G.T, G)
dum2 = alpha_tikh1**2 * np.dot(L1.T, L1)
Ghash = np.dot(la.inv(dum1 + dum2), G.T)

m1 = np.dot(Ghash, t)


# Plot 1sr-order L-curve and add the corner marker.

fig2, ax2 = plt.subplots(1, 1)
ax2.loglog(rho1, eta1)
ax2.loglog(rho_corner1, eta_corner1, 'ro', mfc='None', ms=12, mew=1.5)
ax2.set_xlabel(r'Residual norm $\Vert\textbf{Gm}-\textbf{d}\Vert_{2}$')
ax2.set_ylabel(r'Solution seminorm $\Vert\textbf{Lm}\Vert_{2}$')
fig2.savefig('c4flcurve1.pdf')
plt.close()


# Plot the first-order recovered model and the true model.

fig3, ax3 = plt.subplots(1, 1)
ax3.plot(dobs2, m1*1000, '-', drawstyle='steps')
ax3.plot(depth, strue*1000.0, '--')
ax3.set_xlabel('Depth [m]')
ax3.set_ylabel('Slowness [s/km]')
fig3.savefig('c4fmtikh1.pdf')
plt.close()
