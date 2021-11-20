"""
Copyright (c) 2021 Nima Nooshiri (@nimanzik)
"""

import numpy as np
from scipy import linalg as sla


def solve_svd(G, d, n_keep=None):
    """
    Standard or truncated singular value decomposition (SVD or TSVD)
    solution for system of equations ``Gm=d``.

    Parameters
    ----------
    G : array-like
        Representation of m-by-n system matrix (i.e. forward operator)
    d : array_like
        The data vector.
    n_keep : int (optional)
        Maximum number of singular values used (p). If provided,
        truncates SVD to `n_keep` (TSVD solution). If None (default), all
        singular values are used to obtain the solution.

    Returns
    -------
    m_est : array_like
        The SVD or TSVD solution vector.
    """
    U, s, VT = sla.svd(G, compute_uv=True, full_matrices=True)
    V = np.transpose(VT)
    if n_keep is None:
        n_keep = s.size

    Up = U[:, :n_keep]
    Vp = V[:, :n_keep]
    Sp = np.diag(s[0:n_keep])
    Gdagger = Vp @ sla.inv(Sp) @ Up.T
    m_est = Gdagger @ d
    return m_est


def solve_tikh(G, L, alpha, d):
    """
    Return Tikhonov regularization solution (using the SVD or GSVD)

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
    m_est : array_like
        The GSVD solution vector (regularization solution).
    """
    A = G.T @ G + alpha**2 * np.dot(L.T, L)
    Ghash = sla.inv(A) @ G.T
    m_est = np.dot(Ghash, d)
    return m_est


def solve_tikh_fd(Gspec, Dspec, deltat, order, alpha):
    """
    Return the solution of Tikhonov regularization in frequency domain.

    Parameters
    ----------
    Gspec : array-like of length N
        Discrete Fourier transform of the real-valued array of the
        sampled impulse response **g**, i.e ``Gspec=np.fft.rfft(g)``.

    Dspec : array-like of length N
        Discrete Fourier transform of the real-valued array of the data
        vector **d**, i.e. ``Dspec=np.fft.rfft(d)``.

    deltat : float
        Sampling interval in time/spatial domain.

    order : int, {0, 1, 2}
        The order of the derivative to approximate.

    alpha : float
        The reqularization parameter.

    Returns
    -------
    Mf : array_like
        The model spectrum (regularization solution).
    """
    N = Gspec.size
    if N % 2 == 0:
        ntrans = 2 * (N-1)
    else:
        ntrans = (2*N) - 1

    freqs = np.fft.rfftfreq(ntrans, d=deltat)
    k2p = np.power(2*np.pi*freqs, 2*order, dtype=np.complex)

    numer = np.conj(Gspec) * Dspec
    denom = np.conj(Gspec) * Gspec + np.full_like(numer, alpha*alpha*k2p)
    idx = np.where((np.abs(numer) != 0) & (np.abs(denom) != 0))
    Mf = np.zeros_like(numer, dtype=np.complex)
    Mf[idx] = numer[idx] / denom[idx]
    return Mf


def irls(G, d, eps=1.0e-7, tau=1.0e-7, p=1, n_iter=100):
    """
    Apply Iteratively Reweighted Least Squares (IRLS) method to find an
    approximate p-norm solutions to ``Gm=d``.

    Parameters
    ----------
    G : array-like
        Representation of m-by-n system matrix (i.e. forward operator)
    d : array_like
        The data vector.
    eps : float, optional
        Tolerance below which residuals are ignored (i.e. considered to
        be effectively zero). Ddefault is 1.0e-7.
    tau : float, optional
        Tolerance to stop iteration. The iteration stops when
        ``norm(m_new - m_current) / (1 + norm(m_current)) < tau``.
        Default: 1.0e-7.
    p : int, optional
        Specifies which p-norm to use. Default is 1.
    n_iter : int, optional
        Maximum number of iterations. Default is 100.
    """

    # Initial solution: least-squares solutions
    m_current, *_ = sla.lstsq(G, d, cond=None)

    # Iteratively reweighted system
    n_data = G.shape[0]
    r = np.empty(n_data)
    i_iter = 1

    while i_iter < n_iter:
        i_iter += 1

        # Compute the current residual
        res = d - G @ m_current

        # For each row adjust the weighting factor r based on the residual
        for i_data in range(n_data):
            if (abs_r := np.absolute(res[i_data])) < eps:
                r[i_data] = eps**(p - 2)
            else:
                r[i_data] = abs_r**(p-2)

        # Insert the weighting factors r into matrix R
        R = np.diag(r)

        # Find the solution to the weighted problem
        dummy = G.T @ R
        m_new = sla.inv(dummy @ G) @ (dummy @ d)

        # Check for convergence
        if sla.norm(m_new - m_current, 2) / (1 + sla.norm(m_current)) < tau:
            return m_new
        else:
            m_current = m_new

    print('WARNING: IRLS maximum iterations exceeded.')
    return m_new


__all__ = """
    solve_svd
    solve_tikh
    solve_tikh_fd
    irls
""".split()
