"""
Copyright (c) 2021 Nima Nooshiri (@nimanzik)
"""

import numpy as np


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


__all__ = ['xcorr', 'acorr']
