import math

import numpy as np
from numpy import linalg as la


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


def get_svd_solution(U, s, V, d, nkeep=None):
    """
    Standard or truncated singular value decomposition (SVD or TSVD)
    solution.

    Parameters
    ----------
    U : array_like
        Matrix of data space basis vectors from the SVD.
    s : 1-D array
        Vector of singular values.
    V : array_like
        Matrix of model space basis vectors from the SVD.
    d : array_like
        The data vector.
    nkeep : int (optional)
        Maximum number of singular values used (p). If provided,
        truncates SVD to `nkeep` (TSVD solution).

    Returns
    -------
    m : array_like
        The SVD or TSVD solution vector.
    """

    p = nkeep or s.size
    Up = U[:, 0:p]
    Vp = V[:, 0:p]
    Sp = np.diag(s[0:p])
    Gdagger = np.dot(Vp, np.dot(la.inv(Sp), Up.T))
    m = np.dot(Gdagger, d)
    return m


def get_gsvd_solution(G, L, alpha, d):
    """
    Generalized singular value decomposition (GSVD) solution.

    Parameters
    ----------
    G : array_like
        The system matrix (forward operator or design matrix).
    L : array_like
        The roughening matrix.
    alpha : float
        The reqularization parameter.
    d : array_like
        The data vector.

    Returns
    -------
    m : array_like
        The GSVD solution vector (regularization solution).
    """

    dum1 = np.dot(G.T, G)
    dum2 = alpha**2 * np.dot(L.T, L)
    Ghash = np.dot(la.inv(dum1 + dum2), G.T)
    m = np.dot(Ghash, d)
    return m


__all__ = """
    shaw
    get_svd_solution
    get_gsvd_solution
"""
