#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example 3.2
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
"""

from __future__ import division, print_function

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from peiplib import custompp   # noqa


# ----------------------------------------------------------------------

# --- Set up the basic problem parameters ---

# Time interval, sampling rate etc
tmin, tmax = -5.0, 100.0
dt = 0.5
fs = 1.0 / dt

# Instrument characteristic time constant in [s]
T0 = 10.0

# Noise standard deviation
noise = 0.05

# Discretizing values for `M` and `N` (210 data points)
M = N = 211

# Widths and centers (in [s]) of the true ground acceleration pulses
sig1 = sig2 = 2.0
cntr1, cntr2 = 8.0, 25.0

# Object to save a pdf file with several pages
pdf = PdfPages('ex_8_2.pdf')


# --- Generate time vector ---

t = np.linspace(tmin, tmax, N)


# --- Generate instrument impulse response as a critically-damped pulse ---

g = np.zeros(N-1, dtype=np.float)
for i in range(0, N-1):
    if t[i] > 0:
        g[i] = t[i] * np.exp(-t[i]/T0)

# Normalize instrument response (e.g. max(g)=1)
g0 = np.exp(1)/T0
g *= g0

# --- True signal is two pulses of `sig` standard deviation ---


def afun(amp, cntr, sig, t):
    return amp*np.exp(-(t-cntr)**2 / (2*sig**2))


tt = t[0:N-1]
mtrue = afun(1.0, cntr1, sig1, tt) + afun(0.5, cntr2, sig2, tt)
# The model should be unit height
mtrue /= mtrue.max()

# --- Get the true and noisy data ---

d = np.convolve(g, mtrue, mode='full')
dn = d + noise*np.random.randn(d.size)


# Plot the true model
print('Displaying the true model (fig. 1)')

fig, ax = plt.subplots(1, 1)
ax.plot(tt, mtrue)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')
ax.set_title(r'\textbf{Fig. 1} True model')
pdf.savefig()
plt.close()

# Plot the impulse response
print('Displaying the instrument impulse response (fig. 2)')

fig, ax = plt.subplots(1, 1)
ax.plot(tt, g)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Volt')
ax.set_title(r'\textbf{Fig. 2} Instrument impulse response')
pdf.savefig()
plt.close()

# Plot the noise-free and noisy data
dummy1 = [
    ('noise-free', 3, 'dpred'),
    ('noisy', 4, 'dprednoise')]

for i, y in enumerate([d, dn]):
    dummy2 = dummy1[i]
    print('Displaying the {:s} data (fig. {:d})'.format(*dummy2[:-1]))

    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, y[:tt.size])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Volt')
    ax.set_title(r'\textbf{{Fig. {}}} {} data'.format(
        dummy2[1], dummy2[0].capitalize()))
    pdf.savefig()
    plt.close()


# --- Generate spectra ---


def nextpow2(x):
    return 2**int(np.ceil(np.log2(np.abs(x))))


ntrans = nextpow2(max(d.size, g.size)*1.2)
freqs = np.fft.rfftfreq(ntrans, d=dt)
dspec = np.fft.rfft(d, n=ntrans)
dnspec = np.fft.rfft(dn, n=ntrans)
gspec = np.fft.rfft(g, n=ntrans)

# --- Do the inverse convolution ---
mperf = np.fft.irfft(dspec/gspec, n=ntrans)
mn = np.fft.irfft(dnspec/gspec, n=ntrans)


# Plot the spectra for the impulse response, noise free and noisy data
print('Displaying the spectra for data and impulse response (fig. 5)')

fig, ax = plt.subplots(1, 1)
ax.loglog(freqs, abs(gspec), label='impulse response')
ax.loglog(freqs, abs(dspec), '--', label='noise-free data')
ax.loglog(freqs, abs(dnspec), label='noisy data')
ax.set_xlabel(r'$f$ [Hz]')
ax.set_ylabel('Spectral amplitude')
ax.set_xlim(1.0/N, 1.0)
ax.set_ylim(1.0e-3, 1.0e3)
ax.legend()
ax.set_title(
    r'\textbf{Fig. 5} Amplitude spectra')
pdf.savefig()
plt.close()

# Plot the spectra of the noisy data divided by the spectra of the instrument
print(
    'Displaying the spectra of the noisy data divided by the '
    'instrument (fig. 6)')

fig, ax = plt.subplots(1, 1)
ax.loglog(freqs, abs(dnspec/gspec))
ax.set_xlabel(r'$f$ [Hz]')
ax.set_ylabel('Spectral amplitude')
ax.set_xlim(1.0/N, 1.0)
ax.set_title(r'\textbf{Fig. 6} Spectral division')
pdf.savefig()
plt.close()


# Plot the models recovered from noise-free and noisy data
dummy1 = [
    ('noise-free', 7, 'invconvperf'),
    ('noisy', 8, 'invconvnoise')]

for i, y in enumerate([mperf, mn]):
    dummy2 = dummy1[i]
    print(
        'Displaying the model recovered from the {:s} data '
        '(fig. {:d})'.format(*dummy2[:-1]))

    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, y[:tt.size])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')
    ax.set_title(r'\textbf{{Fig. {}}} Recovered model; {} data'.format(
        dummy2[1], dummy2[0]))
    pdf.savefig()
    plt.close()


# --- Zeroth-order tikhonov regularization ---

# Regularization parameters
emin = -4
emax = 4
nl = 100

alphas = 10**np.linspace(emin, emax, nl)
mod_norm = np.zeros_like(alphas, dtype=np.float)
res_norm = np.zeros_like(alphas, dtype=np.float)
models = np.zeros((ntrans, alphas.size), dtype=np.dtype)

for ialpha, alpha in enumerate(alphas):
    Mf = (gspec.conj()*dspec) / \
        (gspec.conj()*gspec+np.full_like(gspec, alpha**2))

    md = np.fft.irfft(Mf, n=ntrans)

    mod_norm[ialpha] = np.linalg.norm(md)
    res_norm[ialpha] = np.linalg.norm(gspec*Mf - dspec)
    models[:, ialpha] = md


pdf.close()
