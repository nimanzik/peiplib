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


# Plot 1sr-order L-curve and find its corner.

fig2, ax2 = plt.subplots(1, 1)
alpha_tikh1, rho_corn1, eta_corn1 = lcurve_corner(rho1, eta1, reg_params1, ax2)

fig2.savefig('c4flcurve1.pdf')
plt.close()

print('1st-order regularization parameter is:', alpha_tikh1)


# Get the desired model.
dum1 = np.dot(G.T, G)
dum2 = alpha_tikh1**2 * np.dot(L1.T, L1)
Ghash = np.dot(la.inv(dum1 + dum2), G.T)
m1 = np.dot(Ghash, t)


# Plot the first-order recovered model and the true model.

fig3, ax3 = plt.subplots(1, 1)
ax3.plot(depth, strue*1000.0, '--', label='True model')
ax3.plot(dobs2, m1*1000, '-', drawstyle='steps',
        label=r'Recovered model, 1$^{st}$ order reg.')
ax3.set_xlabel('Depth [m]')
ax3.set_ylabel('Slowness [s/km]')
ax3.legend()
fig3.savefig('c4fmtikh1.pdf')
plt.close()



# --- Apply second-order Tikhonov regularization ---

L2 = get_reg_mat(N, 2, full=True)
U2, V2, X2, LAM2, MU2 = gsvd(G, L2)


# Apply the L curve criteria to the first-order regularization problem

rho2, eta2, reg_params2 = lcurve_tikh_gsvd(U2, X2, LAM2, MU2, t, G, L2, 1200)


# Plot 1sr-order L-curve and find its corner.

fig4, ax4 = plt.subplots(1, 1)
alpha_tikh2, rho_corn2, eta_corn2 = lcurve_corner(rho2, eta2, reg_params2, ax4)

fig4.savefig('c4flcurve2.pdf')
plt.close()

print('2nd-order regularization parameter is:', alpha_tikh2)


# Get the desired model.
dum1 = np.dot(G.T, G)
dum2 = alpha_tikh2**2 * np.dot(L2.T, L2)
Ghash = np.dot(la.inv(dum1 + dum2), G.T)
m2 = np.dot(Ghash, t)


# Plot the first-order recovered model and the true model.

fig5, ax5 = plt.subplots(1, 1)
ax5.plot(depth, strue*1000.0, '--', label='True model')
ax5.plot(dobs2, m2*1000, '-', drawstyle='steps',
        label=r'Recovered model, 2$^{nd}$ order reg.')
ax5.set_xlabel('Depth [m]')
ax5.set_ylabel('Slowness [s/km]')
ax5.legend()
fig5.savefig('c4fmtikh2.pdf')
plt.close()
