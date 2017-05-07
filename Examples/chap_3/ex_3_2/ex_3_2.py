#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Example 3.2
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import linalg as la

import custompp
from util import get_cbar_axes


# -----------------------------------------------------------------------------

# --- Set constants ---
# time interval
tmin, tmax = -5, 100

# instrument characteristic time constant in [s]
T0 = 10

# The noise standard deviation
noise = 0.05

# Discretizing values for M & N (210 data points)
M, N = 211, 211

# --- Generate time vector ---
t = np.linspace(tmin, tmax, N)

# --- Generate instrument impulse response as critically-damped pulse ---
g = np.zeros(N-1, dtype=np.float)
for i in range(0, N-1):
    if t[i] > 0:
        g[i] = t[i] * np.exp(-t[i]/T0)

# normalize instrument response (e.g. max(g)=1)
g0 = np.exp(1)/T0
g *= g0

# Plot of instrument response to unit area ground acceleration impulse
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.plot(t[0:N-1], g, 'k')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Volt')
fig0.savefig('c3fimp_resp.pdf')
plt.close()


# --- Populate G matrix ---
G = np.zeros((M-1, N-1), dtype=np.float)
for i in range(1, M):
    for j in range(0, N-1):
        tp = t[i] - t[j]
        if tp > 0:
            G[i-1, j] = tp * np.exp(-tp/T0)

# now divide everything by the denominator
deltat = t[1] - t[0]
G *= (g0 * deltat)

# Display image of G matrix
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cimg = ax1.imshow(G)
cax = get_cbar_axes(ax1)
cbar = fig1.colorbar(cimg, cax=cax)
ax1.set_xlabel('j')
ax1.set_ylabel('i')
fig1.savefig('c3fG.pdf')
plt.close()


# --- Get SVD of G matrix ---
U, S, VT = la.svd(G, compute_uv=True, full_matrices=True)
V = np.transpose(VT)

# Display semilog plot of singular values
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.semilogy(S, 'ko', mfc='None')
ax2.set_xlabel('i')
ax2.set_ylabel(r'$s_{i}$')
ax2.set_xticks(range(0, S.size+1, 50))
fig2.savefig('c3finst_sing.pdf')
plt.close()


# --- True signal is two pulses of sig deviation ---
sig = 2
tt = t[0:N-1]
mtrue = np.exp(-(tt-8)**2 / (2*sig*sig)) + 0.5*np.exp(-(tt-25)**2/(2*sig*sig))
mtrue /= mtrue.max()

# Plot true model
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(tt, mtrue, 'k')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel(r'Acceleration [m/s$^2$]')
fig3.savefig('c3fm_true.pdf')
plt.close()


# --- Get true data without noise ---
d = np.dot(G, mtrue)

# Add random normal noise to the data
dn = d + noise*np.random.randn(M-1)

# Display predicted data using noise-free model
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(tt, d, 'k')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Volt')
fig4.savefig('c3fd_pred.pdf')
plt.close()

# Display predicted data plus random independent noise
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(tt, dn, 'k')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Volt')
fig5.savefig('c3fd_pred_noise.pdf')
plt.close()


# --- Using SVD with all 210 singular values ---
nkeep = N - 1
# Find Up, Sp, Vp
Up = U[:, 0:nkeep]
Vp = V[:, 0:nkeep]
Sp = np.diag(S[0:nkeep])

# Generalized inverse solution for noise-free data (mperf) and noisy data (mn)
Sp_inv = la.inv(Sp)
Gdagger = np.dot(Vp, np.dot(Sp_inv, Up.T))
mperf = np.dot(Gdagger, d)
mn = np.dot(Gdagger, dn)

# Display generalized inverse solution for noise-free data (mperf)
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.plot(tt, mperf, 'k')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel(r'Acceleration [m/s$^2$]')
fig6.savefig('c3fpinv_solution_nonoise.pdf')
plt.close()

# Display generalized inverse solution for noisy data (mn)
fig7 = plt.figure()
ax7 = fig7.add_subplot(111)
ax7.plot(tt, mn, 'k')
ax7.set_xlabel('Time [s]')
ax7.set_ylabel(r'Acceleration [m/s$^2$]')
fig7.savefig('c3fpinv_solution_noise.pdf')
plt.close()


# --- Truncate SVD to 26 singular values ---
nkeep = 26
Up = U[:, 0:nkeep]
Vp = V[:, 0:nkeep]
Sp = np.diag(S[0:nkeep])

# Get model for truncated SVD (m2) with noisy data
Sp_inv = la.inv(Sp)
Gdagger = np.dot(Vp, np.dot(Sp_inv, Up.T))
m2 = np.dot(Gdagger, dn)

# Display TSVD solution
fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
ax8.plot(tt, m2, 'k')
ax8.set_xlabel('Time [s]')
ax8.set_ylabel(r'Acceleration [m/s$^2$]')
fig8.savefig('c3fpinv_solution_noise_26.pdf')
plt.close()


# Get model resolution matrix for p=26
Rm = np.dot(Vp, Vp.T)

# Display image of resolution matrix for TSVD solution
fig9 = plt.figure()
ax9 = fig9.add_subplot(111)
cimg = ax9.imshow(Rm, vmin=Rm.min(), vmax=Rm.max())
cax = get_cbar_axes(ax9)
cbar = fig9.colorbar(cimg, cax=cax)
ax9.set_xlabel('j')
ax9.set_ylabel('i')
fig9.savefig('c3fR_solution_26.pdf')
plt.close()

# Display a column from the model resolution matrix for TSVD solution
fig10 = plt.figure()
ax10 = fig10.add_subplot(111)
ax10.plot(tt, Rm[:, 79], 'k')
ax10.set_xlabel('Time [s]')
ax10.set_ylabel('Element Value')
fig10.savefig('c3fR_column_26.pdf')
plt.close()


# ----- Show successive TSVD solutions -----

p = la.matrix_rank(G)


class UpdateModel(object):
    def __init__(self, ax):
        self.ax = ax
        self.t = tt
        self.m = np.zeros(N-1, dtype=np.float)
        self.res = np.zeros_like(self.m)
        self.mnorm = np.zeros_like(self.m)
        self.title = ''
        self.line, = ax.plot([], [], 'k-')

        # set up plot parameters
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel(r'Acceleration [m/s$^2$]')
        self.ax.set_xlim(tmin, tmax)
        self.ax.set_ylim(-5, 5)

        # plot the true model
        self.ax.plot(self.t, mtrue, 'r--')

    def init(self):
        self.ax.set_title('p = 0')
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):

        if i == 0:
            return self.init()

        # adjust the predicted model to have p singular values (see eq. (3.80))
        dummy = np.dot(U[:, i].T, dn) / S[i]
        self.m += np.dot(dummy, V[:, i])

        # keep track of the residuals for each p
        self.res[i] = la.norm(np.dot(G, self.m) - dn)

        # keep track of the model norm for each p
        self.mnorm[i] = la.norm(self.m)

        # plot the newly fit model
        self.line.set_data(self.t, self.m)

        self.ax.set_title('p = %3d' % (i+1))

        return self.line,


fig11 = plt.figure()
ax11 = fig11.add_subplot(111)
um = UpdateModel(ax11)
anim = FuncAnimation(
    fig11,
    um,
    frames=range(p),
    init_func=um.init,
    interval=50,
    blit=False,
    repeat=False)

plt.show()

# ----- Examine the trade-off curve (collected in the loop above) -----

m = np.zeros(N-1, dtype=np.float)
res = np.zeros_like(m)
mnorm = np.zeros_like(m)

for i in range(p):
    # adjust the predicted model to have p singular values (see eq. (3.80))
    dummy = np.dot(U[:, i].T, dn) / S[i]
    m += np.dot(dummy, V[:, i])

    # keep track of the residuals for each p
    res[i] = la.norm(np.dot(G, m) - dn)

    # keep track of the model norm for each p
    mnorm[i] = la.norm(m)

fig12, ax12 = plt.subplots()
ax12.plot(res, mnorm, 'k')
ax12.plot(res[25], mnorm[25], 'ro', mfc='None')
ax12.set_xlabel(r'Residual norm $\Vert\textbf{Gm}-\textbf{d}\Vert_{2}$')
ax12.set_ylabel(r'Solution norm $\Vert\textbf{m}\Vert_{2}$')
fig12.savefig('c3ftradeoff_curve.pdf')
plt.close()

