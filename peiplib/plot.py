
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


deep_hex = {
    'butter':     '#CCB974',
    'chameleon':  '#55A868',
    'plum':       '#8172B2',
    'scarletred': '#C44E52',
    'skyblue1':   '#64B5CD',
    'skyblue2':   '#4C72B0', }


tango_hex = {
    'butter1':     '#fce94f',
    'butter2':     '#edd400',
    'butter3':     '#c4a000',
    'chameleon1':  '#8ae234',
    'chameleon2':  '#73d216',
    'chameleon3':  '#4e9a06',
    'orange1':     '#fcaf3e',
    'orange2':     '#f57900',
    'orange3':     '#ce5c00',
    'skyblue1':    '#729fcf',
    'skyblue2':    '#3465a4',
    'skyblue3':    '#204a87',
    'plum1':       '#ad7fa8',
    'plum2':       '#75507b',
    'plum3':       '#5c3566',
    'chocolate1':  '#e9b96e',
    'chocolate2':  '#c17d11',
    'chocolate3':  '#8f5902',
    'scarletred1': '#ef2929',
    'scarletred2': '#cc0000',
    'scarletred3': '#a40000',
    'aluminium1':  '#eeeeec',
    'aluminium2':  '#d3d7cf',
    'aluminium3':  '#babdb6',
    'aluminium4':  '#888a85',
    'aluminium5':  '#555753',
    'aluminium6':  '#2e3436', }


tango_RGB = {
    'butter1':     (252, 233,  79),
    'butter2':     (237, 212,   0),
    'butter3':     (196, 160,   0),
    'chameleon1':  (138, 226,  52),
    'chameleon2':  (115, 210,  22),
    'chameleon3':  (78,  154,   6),
    'orange1':     (252, 175,  62),
    'orange2':     (245, 121,   0),
    'orange3':     (206,  92,   0),
    'skyblue1':    (114, 159, 207),
    'skyblue2':    (52,  101, 164),
    'skyblue3':    (32,   74, 135),
    'plum1':       (173, 127, 168),
    'plum2':       (117,  80, 123),
    'plum3':       (92,  53, 102),
    'chocolate1':  (233, 185, 110),
    'chocolate2':  (193, 125,  17),
    'chocolate3':  (143,  89,   2),
    'scarletred1': (239,  41,  41),
    'scarletred2': (204,   0,   0),
    'scarletred3': (164,   0,   0),
    'aluminium1':  (238, 238, 236),
    'aluminium2':  (211, 215, 207),
    'aluminium3':  (186, 189, 182),
    'aluminium4':  (136, 138, 133),
    'aluminium5':  (85,   87,  83),
    'aluminium6':  (46,   52,  54), }


def to01(c):
    return tuple(x/255. for x in c)


def tohex(c):
    return '%02x%02x%02x' % c


def nice_sci_notation(x, ndecimals=2, precision=None, exponent=None):
    """
    https://stackoverflow.com/a/18313780/3202380
    """
    if not exponent:
        exponent = int(math.floor(math.log10(abs(x))))

    coeff = round(x/float(10**exponent), ndecimals)

    precision = precision or ndecimals

    return r"${0:.{1}f}\times10^{{{2:d}}}$".format(coeff, precision, exponent)


def get_cbar_axes(ax, position='right', size='5%', pad='3%'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad=pad)
    return cax


def lcurve(
        rho, eta, ax, reg_c=None, rho_c=None, eta_c=None, flag=0,
        mdf_orig=False, mc=None):
    """
    Plot L-curve (trade-off curve).

    Parameters
    ----------
    rho : array_like
        Vector of residual norm `||Gm-d||_2`.
    eta : array_like
        Vector of solution norm `||m||_2` or seminorm `||LM||_2`.
    ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`
        Set default axes instance.
    reg_c : float (optional)
        The value of ``reg_params`` with maximum curvature.
    rho_c : float (optional)
        The residual norm corresponding to ``reg_c``.
    eta_c : float (optional)
        The solution norm/seminorm corresponding to ``reg_c``.
    flag : int
        Set to 0 (default) for solution norm or 1 for solution seminorm.
    mdf_orig : bool (optional, default=False)
        Set to True if the corner point has been determined by minimum
        distance function (MDF) optimization technique (Belge et al. [2002])
        and plotting origin point described in this technique is desired.
    mc : 3-tuple, 4-tuple, str
        The marker color (one of matplotlib color formats).
    """

    ax.loglog(rho, eta)

    ls = ':'
    lw = 1.0
    lc = tango_hex['aluminium5']
    mc = mc or tango_hex['scarletred2']
    ms = 8

    if mdf_orig:
        ax.annotate(
            "",
            xy=(rho_c, eta_c), xycoords='data',
            xytext=(rho[-1], eta[0]), textcoords='data',
            arrowprops=dict(
                arrowstyle='-|>', connectionstyle='arc3', facecolor='black'),)

        ax.axvline(x=rho[-1], ymax=0.95, linestyle=ls, color=lc, lw=lw)

        ax.axhline(y=eta[0], xmax=0.95, linestyle=ls, color=lc, lw=lw)

        ax.loglog(rho[-1], eta[0], 'ko', ms=ms)

    if rho_c and eta_c:
        l, r = ax.get_xbound()
        b, t = ax.get_ybound()

        ax.loglog(
            [l, rho_c, rho_c], [eta_c, eta_c, b],
            linestyle=ls, color=lc, lw=lw)

        ax.loglog(rho_c, eta_c, 'o', mfc='None', ms=ms, mew=1.5, color=mc)

        if reg_c:
            ax.text(
                1.015*rho_c, 1.15*eta_c,
                r'$\alpha=$%s' % nice_sci_notation(reg_c))

        ax.set_xlim(l, r)
        ax.set_ylim(b, t)

    ax.set_xlabel(r'Residual norm $\Vert\textbf{Gm}-\textbf{d}\Vert_{2}$')

    if flag == 0:
        ax.set_ylabel(r'Solution norm $\Vert\textbf{m}\Vert_{2}$')
    else:
        ax.set_ylabel(r'Solution seminorm $\Vert\textbf{Lm}\Vert_{2}$')


def picard(U, s, d, ax):
    """
    Visual inspection of the discrete Picard condition.

    Parameters
    ----------
    U : array_like
        Matrix of the data space basis vectors from the SVD.
    s : 1-D array
        Vector of singular values.
    d : 1-D array
        The data vector.
    ax : :py:class:`matplotlib.axes._subplots.AxesSubplot`
        Set default axes instance.
    """

    k = s.size
    fcoef = np.zeros(k, dtype=np.float)
    scoef = np.zeros_like(fcoef)
    for i in range(k):
        fcoef[i] = np.dot(U[:, i].T, d)
        scoef[i] = fcoef[i] / s[i]

    x = range(k)
    ax.semilogy(x, s, '-.', label=r'$s_i$')
    ax.semilogy(
        x, fcoef, 'o', label=r'$|\textbf{U}_{.,i}^{T} \textbf{d}|$')
    ax.semilogy(
        x, scoef, 'x', label=r'$|\textbf{U}_{.,i}^{T} \textbf{d}|/s_{i}$')
    ax.legend()
    ax.set_xlabel(r'Index, $i$')
    ax.set_xticks(np.linspace(0, k, 5))


__all__ = """
deep_hex
tango_hex
to01
tohex
nice_sci_notation
get_cbar_axes
lcurve
picard
""".split()
