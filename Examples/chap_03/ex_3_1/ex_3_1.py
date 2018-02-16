#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Example 3.1
Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

import custompp   # noqa
from util import get_cbar_axes


# -----------------------------------------------------------------------------

# Set printing text object separation
sep = '\n'

# --- Dimension of matrix G ---
m = 8
n = 9

# --- Construct system matrix G for the ray path models ---
G = np.zeros((m, n), dtype=np.float)
for i in (0, 1, 2):
    G[i, [i, i+3, i+6]] = 1.0

for i in (3, 4, 5):
    j1 = (i-3) * 3
    j2 = j1 + 3
    G[i, j1:j2] = 1.0

G[6, [0, 4, 8]] = np.sqrt(2)
G[7, -1] = np.sqrt(2)

# --- Get the singular values for the system matrix ---
U, S, VT = la.svd(G, compute_uv=True, full_matrices=True)
V = np.transpose(VT)


# --- Display singular values ---
print('Singular values', 'S =', S.reshape(S.size, 1), '', sep=sep)


# --- Find and display system rank ---
p = la.matrix_rank(G)
print('System rank', 'p =', p, '\n')


# --- Display model null space vectors ---
V0 = V[:, p:n]
print('Model null space vectors', 'V0 =', V0, '', sep=sep)


# --- Display null space vectors reshaped to match tomography example geometry
m01 = np.reshape(V[:, p], (3, 3))
m02 = np.reshape(V[:, p+1], (3, 3))

print(
    'Model null space vectors reshaped into matrices',
    'm01 =', m01, '', 'm02 =', m02, '', sep=sep)


# --- Display image of null space model V.,8 ---
fig0 = plt.figure()
ax = fig0.add_subplot(111)
cimg = ax.imshow(m01)
cax = get_cbar_axes(ax)
cbar = fig0.colorbar(cimg, ticks=[-0.5, 0.0, 0.5], cax=cax)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig0.savefig('c3fv8null.pdf')


# --- Display image of null space model V.,8 ---
fig1 = plt.figure()
ax = fig1.add_subplot(111)
cimg = ax.imshow(m02)
cax = get_cbar_axes(ax)
cbar = fig1.colorbar(cimg, cax=cax, ticks=[-0.5, 0.0, 0.5])
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig1.savefig('c3fv9null.pdf')


# --- Display data null space vectors ---
U0 = U[:, p:]
print('Data null space vector', 'U0 =', U0, '', sep=sep)


# --- Find and display model resolution matrix ---
Vp = V[:, 0:p-1]
Rm = np.dot(Vp, Vp.T)


# --- Display full model resolution matrix ---
fig2 = plt.figure()
ax = fig2.add_subplot(111)
cimg = ax.imshow(Rm, vmin=-0.1, vmax=1.0)
cax = get_cbar_axes(ax)
cbar = fig2.colorbar(cimg, cax=cax, ticks=np.arange(0, 1.2, 0.2))
ax.set_xticks(range(9))
ax.set_yticks(range(9))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig2.savefig('c3fR.pdf')


# --- Display reshaped diagonal elements of the model resolution matrix ---
Rm_diag = np.diagonal(Rm).reshape(3, 3)
print(
    'Diagonal elements of model resolution matrix',
    'Rm_diag=', Rm_diag, '', sep=sep)

fig3 = plt.figure()
ax = fig3.add_subplot(111)
cimg = ax.imshow(Rm_diag, vmin=-0.1, vmax=1.0)
cax = get_cbar_axes(ax)
cbar = fig3.colorbar(cimg, cax=cax, ticks=np.arange(0, 1.2, 0.2))
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig3.savefig('c3fRdiag.pdf')


# --- Spike resolution test ---
# --- Construct spike model ---
mtest = np.zeros((n, 1))
mtest[4, 0] = 1.0

# --- Get noise free data for the spike model (forward problem) ---
dtest = np.dot(G, mtest)

# --- Display spike model and noise-free data
print(
    'Model spike and predicted data',
    'mtest =', mtest, '', 'dtest =', dtest, '', sep=sep)

# --- Display recovered model from spark test ---
Gdagger = la.pinv(G)
mdagger = np.dot(Gdagger, dtest)

print(
    'Recovered model from the spike test',
    'mdagger =', mdagger, '', sep=sep)


# --- Display reshaped noise free spike model ---
fig4 = plt.figure()
ax = fig4.add_subplot(111)
cimg = ax.imshow(np.reshape(mtest, (3, 3)), vmin=-0.1, vmax=1.0)
cax = get_cbar_axes(ax)
cbar = fig4.colorbar(cimg, cax=cax, ticks=np.arange(0, 1.2, 0.2))
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig4.savefig('c3fspike.pdf')


# --- Display reshaped recovered spike model
fig5 = plt.figure()
ax = fig5.add_subplot(111)
cimg = ax.imshow(np.reshape(mdagger, (3, 3)), vmin=-0.1, vmax=1.0)
cax = get_cbar_axes(ax)
cbar = fig5.colorbar(cimg, cax=cax, ticks=np.arange(0, 1.2, 0.2))
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xlabel('j')
ax.set_ylabel('i')
fig5.savefig('c3fspike_recov.pdf')
