import numpy as np
import numpy.linalg as nla

from peiplib.plot import nice_sci_notation
from peiplib.util import loglinspace


class UpdateTikhonovModelFrequncy(object):

    def __init__(
            self, ax, xdata, Gspec, Dspec, deltat, order, npoints,
            alpha_min, alpha_max):
        self.ax = ax
        self.xdata = xdata
        self.Gspec = Gspec
        self.Dspec = Dspec
        self.deltat = deltat
        self.order = order
        self.alphas = loglinspace(alpha_min, alpha_max, npoints)
        if Gspec.size % 2 == 0:
            self.ntrans = 2 * (Gspec.size-1)
        else:
            self.ntrans = (2*Gspec.size) - 1
        self.freqs = np.fft.rfftfreq(self.ntrans, d=self.deltat)
        self.ydata = np.zeros((self.alphas.size, self.ntrans), dtype=np.float)
        self.rnorm = np.zeros(npoints, dtype=np.float)
        self.mnorm = np.zeros(npoints, dtype=np.float)
        self.__GHD = np.conj(Gspec) * Dspec
        self.__GHG = np.conj(Gspec) * Gspec
        self.__k2p = np.power(2*np.pi*self.freqs, 2*order)
        self.line, = ax.plot([], [], 'k-')

    def init_func(self):
        self.ax.set_title(r'$\alpha$={}'.format(self.alphas[0]))
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):

        alpha = self.alphas[i]

        # Predicted model; freq domain
        Mf = self.__GHD / \
            (self.__GHG + np.full_like(self.__GHG, alpha*alpha*self.__k2p))

        # Predicted model; time domain
        md = np.fft.irfft(Mf, n=self.ntrans)

        # Store predicted model for each alpha
        self.ydata[i, :] = md

        # Keep track of the residual norm for each alpha
        self.rnorm[i] = nla.norm(self.Gspec*Mf-self.Dspec)

        # Keep track of the model norm for each alpha
        self.mnorm[i] = nla.norm(Mf)

        # Plot the newly fit model
        self.line.set_data(self.xdata, md[:len(self.xdata)])
        self.ax.set_title(r'$\alpha$={}'.format(nice_sci_notation(alpha)))

        return self.line,
