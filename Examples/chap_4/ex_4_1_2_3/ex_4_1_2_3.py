#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Example 4.1-2-3
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy.interpolate import interp1d
from scipy.sparse.linalg import lsqr

from peiplib import custompp
from peiplib.reg import lcurve_corner, lcurve_tikh_svd
from peiplib.util import get_cbar_axes, picard, shaw


pi = np.pi

# -----------------------------------------------------------------------------

# Load in the Shaw problem for m=n=20
infile = np.load('shawexamp.npz')
G = infile['G']
dspike = infile['dspike']
dspiken = infile['dspiken']
theta = infile['theta']

noise = 1.0e-6
dspiken = dspike + noise*np.random.randn(dspike.size)


# --- Example 4.1 ---

# Calculate the SVD
U, s, VT = la.svd(G)
V = np.transpose(VT)


# --- First, calculate and plot the L-curve, and find its corner ---
rho, eta, reg_params = lcurve_tikh_svd(U, s, dspiken, 1000)

# Estimate the L-curve corner in log-log space
alpha_tikh, icorner, _ = lcurve_corner(rho, eta, reg_params)
rho_corner = rho[icorner]
eta_corner = eta[icorner]

print('alpha from the L-curve is:', alpha_tikh)


# --- Get the spike solution corresponding to the L-curve corner ---
M, N = G.shape
dum1 = np.dot(G.T, G) + (alpha_tikh**2)*np.identity(N, dtype=np.float)
dum2 = np.dot(G.T, dspiken)
m_tikh = la.solve(dum1, dum2)


# Get the residual using the L-curve solution
r_spike = la.norm(np.dot(G, m_tikh) - dspiken)

print('Residual norm for L-curve solution using Tikhonov reg.:', r_spike)


# Plot the L-curve and add the corner marker
fig0, ax0 = plt.subplots(1, 1)
ax0.loglog(rho, eta)
ax0.loglog(rho_corner, eta_corner, 'ro', mfc='None', ms=12, mew=1.5)
ax0.set_xlabel(r'Residual norm $\Vert\textbf{Gm}-\textbf{d}\Vert_{2}$')
ax0.set_ylabel(r'Solution norm $\Vert\textbf{m}\Vert_{2}$')
fig0.savefig('c4flcurve0.pdf')
plt.close()


# Plot the L-curve predicted model
fig1, ax1 = plt.subplots(1, 1)
ax1.step(theta, m_tikh, where='mid')
ax1.set_xlabel(r'$\theta$ [rad]')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-0.25, 0.5)
fig1.savefig('c4fmtikh.pdf')
plt.close()


# --- Use the discrepancy principle to get a second solution ---

# find the regularization value, alpha_disc, for rho=discrep by interpolation
# of the L-curve
discrep = np.sqrt(20.) * noise
f = interp1d(rho, reg_params)
alpha_disc = f(discrep)

print('alpha from the discrepancy principle is:', alpha_disc)

# Get the model and residual
m_disc = lsqr(G, dspiken, damp=alpha_disc)[0]
r_spike_disc = la.norm(np.dot(G, m_disc) - dspiken)

print(
    'Residual norm for discrepancy principle solution using Tikhonov reg.:',
    r_spike_disc)

# Plot the discrepancy principle predicted model
fig2, ax2 = plt.subplots(1, 1)
ax2.step(theta, m_disc, where='mid')
ax2.set_xlabel(r'$\theta$ [rad]')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-0.25, 0.5)
fig2.savefig('c4fmdisc.pdf')
plt.close()


# --- Picard plot ---
fig3, ax3 = plt.subplots(1, 1)
picard(U, s, dspiken, ax3)
fig3.savefig('c4fpicard.pdf')
plt.close()


# --- Example 4.2 ---

# Now, examine the resolution using a noise-free spike test for alpha_tikh
spikemod_tikh = lsqr(G, dspike, damp=alpha_tikh)[0]

# Plot the noise-free spike model for the L-curve
fig4, ax4 = plt.subplots(1, 1)
ax4.step(theta, spikemod_tikh, where='mid')
ax4.set_xlabel(r'$\theta$ [rad]')
ax4.set_ylabel('Intensity')
ax4.set_xlim(-2, 2)
ax4.set_ylim(-0.25, 0.5)
fig4.savefig('c4fpinv_spike_nonoise_tikh.pdf')


# Now, examine the resolution using a noise-free spike test for alpha_disc
spikemod_disc = lsqr(G, dspike, damp=alpha_disc)[0]

# Plot the noise-free spike model for the discrepancy principle
fig5, ax5 = plt.subplots(1, 1)
ax5.step(theta, spikemod_disc, where='mid')
ax5.set_xlabel(r'$\theta$ [rad]')
ax5.set_ylabel('Intensity')
ax5.set_xlim(-2, 2)
ax5.set_ylim(-0.25, 0.5)
fig5.savefig('c4fpinv_spike_nonoise_disc.pdf')


# --- Compute the resolution matrix for alpha_disc ---
M, N = G.shape
dummy = np.dot(G.T, G) + (alpha_disc**2)*np.identity(N, dtype=np.float)
Ghash = np.dot(la.inv(dummy), G.T)
Rm_disc = np.dot(Ghash, G)

print('Diagonal resolution elements:\n', np.diag(Rm_disc))

# Plot the resolution matrix
fig6, ax6 = plt.subplots(1, 1)
cimg = ax6.imshow(Rm_disc, vmin=-0.1, vmax=1.0)
cax = get_cbar_axes(ax6)
cbar = fig6.colorbar(cimg, cax=cax)
ax6.set_xlabel('j')
ax6.set_ylabel('i')
fig6.savefig('c4fshaw_res.pdf')


# --- Example 4.3 ---

# Get the covariance of the discrepancy principle solution
dummy = np.dot(Ghash, (noise**2)*np.identity(N, dtype=np.float))
covm_disc = np.dot(dummy, Ghash.T)
conf95 = 1.96 * np.sqrt(np.diag(covm_disc))


# Plot of the discrepancy principle solution with error bars vs. reality
spike = np.zeros(N, dtype=np.float)
spike[9] = 1.0

dtheta = theta[1] - theta[0]
theta2 = theta + dtheta/2

fig7, ax7 = plt.subplots(1, 1)
ax7.plot(theta2, spike, drawstyle='steps', label=r'$\textbf{m}_{true}$')
ax7.plot(theta2, m_disc, '--', drawstyle='steps', label=r'$\textbf{m}_{disc.}$')
dum = ax7.plot(theta2, m_disc+conf95, ':', drawstyle='steps', label=r'95$\%$')
color = plt.get(dum[0], 'color')
ax7.plot(theta2, m_disc-conf95, ':', color=color, drawstyle='steps')
ax7.set_xlabel(r'$\theta$ [rad]')
ax7.set_ylabel('Intensity')
ax7.legend()
fig7.savefig('c4fbias.pdf')
