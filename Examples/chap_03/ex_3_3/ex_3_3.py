#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Example 3.3
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

import custompp   # noqa
from util import shaw, tsvd_solution


# -----------------------------------------------------------------------------

# --- Set constants ---
# Discretizing values for M & N (20 data points)
M, N = 20, 20

# The noise standard deviation
noise = 1.0e-6


# --- Get the data for m=n=20 ---
gamma, theta, G = shaw(M, N)


# --- Compute the SVD ---
U, S, VT = la.svd(G, full_matrices=True, compute_uv=True)
V = np.transpose(VT)

# Display semilog plot of singular values
fig0, ax0 = plt.subplots(1, 1)
ax0.semilogy(S, 'ko', mfc='None')
ax0.set_xlabel('i')
ax0.set_ylabel(r'$s_{i}$')
ax0.set_xticks(range(0, S.size+1, 5))
fig0.savefig('c3fshaw_sing.pdf')
plt.close()


# --- Plot column of V corresponds to the smallest nonzero singular value ---
p = la.matrix_rank(G)
print('M =', M, 'and', 'N =', N)
print('System rank', 'p =', p, '\n')

fig1, ax1 = plt.subplots(1, 1)
ax1.step(theta, V[:, p-1], 'k-', where='mid')
ax1.set_xlabel(r'$\theta$ [rad]')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-2, 2)
fig1.savefig('c3fV_18.pdf')
plt.close()


# --- Plot column of V corresponds to the largest nonzero singular value ---
fig2, ax2 = plt.subplots(1, 1)
ax2.step(theta, V[:, 0], 'k-', where='mid')
ax2.set_xlabel(r'$\theta$ [rad]')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-2, 2)
fig2.savefig('c3fV_1.pdf')
plt.close()


# --- Create a spike model ---
spike = np.zeros(N, dtype=np.float)
spike[9] = 1.0

# --- Get the ideal Shaw spike data ---
dspike = np.dot(G, spike)

# Displaying the spike model
fig3, ax3 = plt.subplots(1, 1)
ax3.step(theta, spike, 'k', where='mid')
ax3.set_xlabel(r'$\theta$ [rad]')
ax3.set_ylabel('Intensity')
ax3.set_xlim(-2, 2)
ax3.set_ylim(-0.25, 1.05)
fig3.savefig('c3fshawspike.pdf')
plt.close()

# Displaying noise-free Shaw spike data
fig4, ax4 = plt.subplots(1, 1)
ax4.step(theta, dspike, 'k', where='mid')
ax4.set_xlabel(r'$\theta$ [rad]')
ax4.set_ylabel('Intensity')
ax4.set_xlim(-2, 2)
ax4.set_ylim(-0.25, 1.05)
fig4.savefig('c3fshawspike_data_nonoise.pdf')
plt.close()


# --- Create slightly noisy data (dspiken) ---
dspiken = dspike + noise*np.random.randn(M)

# --- Generalized solution for noise-free data ---
spikemod = la.solve(G, dspike)

# --- Find the pseudoinverse solution with noisy data for p=18 ---
nkeep = p
spikemod18n = tsvd_solution(U, S, V, nkeep, dspiken)


# --- Plot generalized inverse solution for noise-free spike model data ---
fig5, ax5 = plt.subplots(1, 1)
ax5.step(theta, spikemod, 'k-', where='mid')
ax5.set_xlabel(r'$\theta$ [rad]')
ax5.set_ylabel('Intensity')
ax5.set_xlim(-2, 2)
ax5.set_ylim(-0.25, 1.05)
fig5.savefig('c3fpinv_spike_nonoise.pdf')
plt.close()


# --- Plot recovered model for noisy data ---
fig6, ax6 = plt.subplots(1, 1)
ax6.step(theta, spikemod18n, 'k-', where='mid')
ax6.set_xlabel(r'$\theta$ [rad]')
ax6.set_ylabel('Intensity')
ax6.set_xlim(-2, 2)
ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
fig6.savefig('c3fpinv_spike_noise_18.pdf')
plt.close()


# --- Find the pseudoinverse solution with noisy data for p=10 ---
nkeep = 10

# recover the noise-free model
spikemod10 = tsvd_solution(U, S, V, nkeep, dspike)
# recover the noisy model
spikemod10n = tsvd_solution(U, S, V, nkeep, dspiken)


# Plot recovered model for noise-free data, p=10
fig7, ax7 = plt.subplots(1, 1)
ax7.step(theta, spikemod10, 'k-', where='mid')
ax7.set_xlabel(r'$\theta$ [rad]')
ax7.set_ylabel('Intensity')
ax7.set_xlim(-2, 2)
ax7.set_ylim(-0.25, 1.05)
fig7.savefig('c3fpinv_spike_nonoise_10.pdf')
plt.close()


# Plot recovered model for noisy data, p=10
fig8, ax8 = plt.subplots(1, 1)
ax8.step(theta, spikemod10n, 'k-', where='mid')
ax8.set_xlabel(r'$\theta$ [rad]')
ax8.set_ylabel('Intensity')
ax8.set_xlim(-2, 2)
ax8.set_ylim(-0.25, 1.05)
fig8.savefig('c3fpinv_spike_noise_10.pdf')
plt.close()


# -----------------------------------------------------------------------------

# --- Get the data for m=n=100 ---
M2, N2 = 100, 100
gamma2, theta2, G2 = shaw(M2, N2)

# Compute the SVD
p2 = la.matrix_rank(G2)
print('M =', M2, 'and', 'N =', N2)
print('System rank', 'p =', p2, '\n')

U2, S2, VT2 = la.svd(G2, full_matrices=True, compute_uv=True)
V2 = np.transpose(VT2)

# Get spike for n=100
spike2 = np.zeros(N2, dtype=np.float)
spike2[45:50] = 1.0

# Get spike data for n=100 case without noise
dspike2 = np.dot(G2, spike2)
# add noise to get noisy data
dspiken2 = dspike2 + noise*np.random.randn(M2)

# Recover the model from noisy data
nkeep = 10
spikemod10n_2 = tsvd_solution(U2, S2, V2, nkeep, dspiken2)


# --- Plot spectrum of singular values for n=100 problem ---
fig9, ax9 = plt.subplots(1, 1)
ax9.semilogy(S2, 'ko', mfc='None')
ax9.set_xlabel('i')
ax9.set_ylabel(r'$s_{i}$')
ax9.set_xticks(range(0, S2.size+1, 20))
ax9.set_ylim(1.0e-20, 1.0e5)
fig9.savefig('c3fshaw_sing_100.pdf')
plt.close()


# --- Plot recovered model for noisy data with n=100, p=10 ---
fig10, ax10 = plt.subplots(1, 1)
ax10.step(theta2, spikemod10n_2, 'k-', where='mid')
ax10.set_xlabel(r'$\theta$ [rad]')
ax10.set_ylabel('Intensity')
ax10.set_xlim(-2, 2)
ax10.set_ylim(-0.25, 1.05)
fig10.savefig('c3fpinv_spike_noise_100_10.pdf')
plt.close()


# --- Now try p=18 case on n=100 data ---
nkeep = 18
spikemod18n_2 = tsvd_solution(U2, S2, V2, nkeep, dspiken2)

# --- Plot recovered model for noisy data with n=100, p=18 ---
fig11, ax11 = plt.subplots(1, 1)
ax11.step(theta2, spikemod18n_2, 'k-', where='mid')
ax11.set_xlabel(r'$\theta$ [rad]')
ax11.set_ylabel('Intensity')
ax11.set_xlim(-2, 2)
ax11.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
fig11.savefig('c3fpinv_spike_noise_100_18.pdf')
plt.close()


# -----------------------------------------------------------------------------

# --- Plot singular values of G for n=6 ---
M3, N3 = 6, 6
_, _, G3 = shaw(M3, N3)

# Compute the SVD
p3 = la.matrix_rank(G3)
print('M =', M3, 'and', 'N =', N3)
print('System rank', 'p =', p3, '\n')

S3 = la.svd(G3, compute_uv=False)

fig12, ax12 = plt.subplots(1, 1)
ax12.semilogy(S3, 'ko', mfc='None')
ax12.set_xlabel('i')
ax12.set_ylabel(r'$s_{i}$')
ax12.set_xticks(range(0, S3.size+1))
ax12.set_ylim(1.0e-2, 1.0e1)
fig12.savefig('c3fshaw_sing_6.pdf')
plt.close()


# Save arrays that are use later
np.savez_compressed(
    'shawexamp.npz',
    G=G,
    dspike=dspike,
    dspiken=dspiken,
    theta=theta,
    gamma=gamma)
