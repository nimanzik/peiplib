import math

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import linalg as la


pi = math.pi
sin = math.sin
cos = math.cos


def shaw(m, n):
    '''
    This function generates and returns outgoing angles vector `gamma` (see
    eq. (1.40)), incoming angles vactor `theta` (see eq. (1.41)), and forward
    operator G (see eq. (1.44)) for Shaw problem.

    Parameters
    ----------
    m: int
        Number of data points.
    n: int
        Number of model points.

    Returns
    -------
    gamma: ndarray
        Data vector, `m` equally spaced outgoing angles.
    theta: ndarray
        Model vector, `n` equally spaced incoming angles.
    G: ndarray
        Operator G, a 2-D array with shape `m*n`.
    '''
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
    '''
    Standard or truncated singular value decomposition (SVD or TSVD) solution.

    Parameters
    ----------
    U: ndarray
        Matrix of data space basis vectors from the SVD.
    s: 1-D array
        Vector of singular values.
    V: ndarray
        Matrix of model space basis vectors from the SVD.
    d: ndarray
        The data vector.
    nkeep: int (optional)
        Maximum number of singular values used (p). If provided, truncates SVD
        to `nkeep` (TSVD solution).

    Returns
    -------
    m: ndarray
        The SVD or TSVD solution vector.
    '''
    p = nkeep or s.size
    Up = U[:, 0:p]
    Vp = V[:, 0:p]
    Sp = np.diag(s[0:p])
    Gdagger = np.dot(Vp, np.dot(la.inv(Sp), Up.T))
    m = np.dot(Gdagger, d, dtype=np.float)
    return m


def get_gsvd_solution(G, L, alpha, d):
    '''Generalized singular value decomposition (GSVD) solution.

    Parameters
    ----------
    G: ndarray
        The system matrix (forward operator or design matrix).
    L: ndarray
        The roughening matrix.
    alpha: float
        The reqularization parameter.
    d: ndarray
        The data vector.

    Returns
    -------
    m: ndarray
        The GSVD solution vector (regularization solution).
    '''
    dum1 = np.dot(G.T, G)
    dum2 = alpha**2 * np.dot(L.T, L)
    Ghash = np.dot(la.inv(dum1 + dum2), G.T)
    m = np.dot(Ghash, d, dtype=np.float)
    return m


def picard(U, s, d, ax):
    '''Visual inspection of the discrete Picard condition.

    Parameters
    ----------
    U: ndarray
        Matrix of the data space basis vectors from the SVD.
    s: 1-D array
        Vector of singular values.
    d: 1-D array
        The data vector.
    ax: :py:class:``matplotlib.axes._subplots.AxesSubplot`` instance
        Set default axes instance.
    '''
    k = s.size
    fcoef = np.zeros(k, dtype=np.float)
    scoef = np.zeros_like(fcoef)
    for i in range(k):
        fcoef[i] = np.dot(U[:, i].T, d)
        scoef[i] = fcoef[i] / s[i]

    x = range(k)
    ax.semilogy(x, s, '-.', label=r'$s_i$')
    ax.semilogy(x, fcoef, 'o', label=r'$|\textbf{U}_{.,i}^{T} \textbf{d}|$')
    ax.semilogy(x, scoef, 'x', label=r'$|\textbf{U}_{.,i}^{T} \textbf{d}|/s_{i}$')
    ax.legend()
    ax.set_xlabel('i')
    ax.set_xticks(np.linspace(0, k, 5))


def get_cbar_axes(ax, position='right', size='5%', pad='3%'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad=pad)
    return cax
