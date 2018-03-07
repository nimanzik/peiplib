import numpy as np

from peiplib.plot import nice_sci_notation, tango_hex


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
lcurve
picard
""".split()
