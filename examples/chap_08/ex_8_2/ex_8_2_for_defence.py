#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example 3.2
from Parameter Estimation and Inverse Problems, 2nd edition, 2011
by R. Aster, B. Borchers, C. Thurber

:author: Nima Nooshiri (nima.nooshiri@gfz-potsdam.de)
"""

from __future__ import division, print_function

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from peiplib import custompp   # noqa
from peiplib import tikhonov as tikh
from peiplib.plot import lcurve, tango_hex as tango

from test_updatemodel import UpdateTikhonovModelFrequncy


# ----------------------------------------------------------------------

# ----- Set up the basic problem parameters -----

# Time interval, sampling rate etc
tmin, tmax = -5.0, 55.0
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
pdf = PdfPages('ex_8_2_for_defence.pdf')


# ----- Generate time vector -----

t = np.linspace(tmin, tmax, N)


# ----- Generate instrument impulse response as a critically-damped pulse ---

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

# ----- Get the true and noisy data -----

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


# ----- Generate spectra -----


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


# ----- 2nd-order Tikhonov regularization -----

npoints = 250
alpha_min = 0.001
alpha_max = 100

# Animation of evolving solution (not saved)
fig, ax = plt.subplots(1, 1)
ax.plot(tt, mtrue, '--', c=tango['skyblue2'], lw=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')

utm_t2 = UpdateTikhonovModelFrequncy(
    ax, tt, gspec, dnspec, dt, 2, npoints, alpha_min, alpha_max)

anim_t2 = FuncAnimation(
    fig, utm_t2, frames=npoints, init_func=utm_t2.init_func,
    interval=10, blit=False, repeat=False)
plt.show()

# Plot the L-curve with respect to zeroth-order regularization
print(
    'Displaying the L-curve for 2nd-order Tikhonov regularization (fig. 12)')

alphac_t2, rhoc_t2, etac_t2 = tikh.lcorner_kappa(
    utm_t2.rnorm, utm_t2.mnorm, utm_t2.alphas)

fig, ax = plt.subplots(1, 1)
lcurve(
    utm_t2.rnorm, utm_t2.mnorm, ax, reg_c=alphac_t2, rho_c=rhoc_t2,
    eta_c=etac_t2, freqdomain=True, seminorm=True)

ax.set_title(r'\textbf{Fig. 12} 2$^{nd}$-order Tikhonov L-curve')
pdf.savefig()
plt.close()

# Plot the suite of solutions
print('Displaying the suite of 2nd-order regularized models (fig. 13)')

scale_factor = 0.2

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
colors = (tango['aluminium6'], tango['scarletred2'])
linewidths = (1.5, 2.5)

step = 35
for i in range(0, npoints, step):
    alpha = utm_t2.alphas[i]
    vshift = np.log10(alpha)
    x = utm_t2.xdata
    y = utm_t2.ydata[i, 0:x.size]

    ax.plot(x, vshift+scale_factor*mtrue, '--', c=tango['skyblue2'])

    j = int(alpha == alphac_t2)
    if i == step*4:
        color = colors[1]
    else:
        color = colors[j]
    ax.plot(x, vshift+scale_factor*y, c=color, lw=linewidths[j])

# Highlight the selected solution
vshift = np.log10(alphac_t2)
idx = np.where(utm_t2.alphas == alphac_t2)

# ax.plot(
#     tt, vshift+scale_factor*mtrue, '--', c=tango['skyblue2'],
#     lw=linewidths[1])

# ax.plot(
#     utm_t2.xdata,
#     vshift+scale_factor*utm_t2.ydata[idx].ravel()[:utm_t2.xdata.size],
#     c=colors[1],
#     lw=linewidths[1])

ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Log$_{10}(\alpha)$')
ax.set_xticks([])
# ax.set_title(r'\textbf{Fig. 13} 2$^{nd}$-order Tikhonov models')
fig.savefig('Lcurve_models.pdf')
fig.savefig('Lcurve_models.png', dpi=300)
pdf.savefig()
plt.close()

# Plot the best solution from zeroth-order Tikhonov regularization
print('Displaying the preferred 2nd-order regularized model (fig. 14)')

fig, ax = plt.subplots(1, 1)
ax.plot(tt, mtrue, '--', c=tango['skyblue2'])
ax.plot(utm_t2.xdata, utm_t2.ydata[idx].ravel()[:utm_t2.xdata.size])
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Acceleration [$\frac{m}{s^2}$]')
ax.set_title(r'\textbf{Fig. 14} Preferred 2$^{nd}$-order Tikhonov model')
pdf.savefig()
plt.close()

pdf.close()
