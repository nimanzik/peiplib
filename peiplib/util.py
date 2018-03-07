import math

import numpy as np


pi = math.pi
sin = math.sin
cos = math.cos


def loglinspace(start, stop, num):
    """
    Returns evenly spaced values in logarithmic scale.

    The values are evenly spaced over log10(start) and log10(stop)
    interval.
    """
    return np.logspace(
        np.log10(start), np.log10(stop), num=num, endpoint=True, base=10.)


def nextpow2(x):
    return 2**int(np.ceil(np.log2(np.abs(x))))


def isvector(x):
    if (x.ndim > 2) or (x.ndim == 2 and x.shape[1] != 1):
        return False
    return True


def xcorr(x, y, maxlag=None, normed=False, want_lags=False):
    """
    Cross-corellation between two 1-D sequences *x* and *y*

    Parameters
    ----------
    x, y : array-like
        Input sequences of scalars of length N.

    maxlag : int (optional, default=None)
        Maximum lag (i.e. number of lags to show). Limits the lag range
        from ``-maxlag`` to ``maxlag``. If ``None``, the lag range
        equals all ``2N - 1`` lags.

    normed : bool (optional, default: False)
        if ``True``, input sequences are normalized to unit length (i.e.
        normalizes the cross-correlation sequence so that the
        auto-correlations at zero lag equal 1).

    want_lags : bool (optional, default: False)
        Whether to return the vector of lag indices in output.

    Returns
    -------
    cc : array-like (length ``(2 * maxlag) + 1``)
        Cross-correlation sequence.

    lags : array-like (optional, length ``(2 * maxlag) + 1``)
        Lag indices, which is returnd in output if ``want_lags`` is set
        to ``True``.

    Notes
    -----
    The cross-correlation is performed with :func:`numpy.correlate` with
    ``mode='full'``.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    if len(y) != nx:
        raise ValueError('input sequences x and y must be the same length')

    if maxlag is None:
        maxlag = nx - 1

    if maxlag >= nx or maxlag < 1:
        raise ValueError(
            'maxlag must be None or strictly a positive '
            'integer < {:d}'.format(nx))

    cc = np.correlate(x, y, mode='full')

    if normed:
        Rxx = np.correlate(x, x, mode='valid')
        Ryy = np.correlate(y, y, mode='valid')
        cc /= np.sqrt(Rxx*Ryy)

    cc = cc[nx-1-maxlag:nx+maxlag]

    if want_lags:
        lags = np.arange(-maxlag, maxlag+1)
        return cc, lags
    return cc


def acorr(x, **kwargs):
    """
    Auto-correlation of 1-D sequence *x*.

    Parameters
    ----------
    x : array-like
        Input sequence of scalar of length N.

    maxlag : int (optional, default=None)
        Maximum lag (i.e. number of lags to show). Limits the lag range
        from ``-maxlag`` to ``maxlag``. If ``None``, the lag range
        equals all ``2N - 1`` lags.

    normed : bool (optional, default: False)
        if ``True``, input sequences are normalized to unit length (i.e.
        normalizes the cross-correlation sequence so that the
        auto-correlations at zero lag equal 1).

    want_lags : bool (optional, default: False)
        Whether to return the vector of lag indices in output.

    Returns
    -------
    ac : array-like (length ``(2 * maxlag) + 1``)
        Auto-correlation sequence.

    lags : array-like (optional, length ``(2 * maxlag) + 1``)
        Lag indices, which is returnd in output if ``want_lags`` is set
        to ``True``.

    Notes
    -----
    The cross-correlation is performed with :func:`numpy.correlate` with
    ``mode='full'``.
    """
    return xcorr(x, x, **kwargs)


def shaw(m, n):
    """
    This function generates and returns outgoing angles vector `gamma`
    (see eq. (1.40)), incoming angles vactor `theta` (see eq. (1.41)),
    and forward operator G (see eq. (1.44)) for Shaw problem.

    Parameters
    ----------
    m : int
        Number of data points.
    n : int
        Number of model points.

    Returns
    -------
    gamma : array_like
        Data vector, `m` equally spaced outgoing angles.
    theta : array_like
        Model vector, `n` equally spaced incoming angles.
    G : array_like
        Operator G, a 2-D array with shape `m-by-n`.
    """

    i = np.arange(1, m+1)
    gamma = (i-0.5)*pi/m - pi/2

    j = np.arange(1, n+1)
    theta = (j-0.5)*pi/n - pi/2

    dtheta = pi/n
    G = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            gamma_i = gamma[i]
            theta_j = theta[j]
            dum1 = cos(gamma_i) + cos(theta_j)
            dum2 = pi * (sin(gamma_i) + sin(theta_j))
            try:
                G[i, j] = dum1**2 * (sin(dum2)/dum2)**2 * dtheta
            except ZeroDivisionError:
                G[i, j] = dum1**2 * 1.0 * dtheta

    return (gamma, theta, G)


__all__ = """
    loglinspace
    nextpow2
    xcorr
    acorr
    shaw
""".split()
