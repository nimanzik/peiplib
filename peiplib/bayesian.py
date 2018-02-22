import numpy as np
from numpy import linalg as nla
from scipy import linalg as sla

from peiplib.util import acorr


def bayes(G, mprior, covm, d, covd):
    """
    Compute maximum a posteriori (MAP) solution and corresponding
    posterior covariance matrix.

    Parameters
    ----------
    G : array-like
        The system matrix (i.e. forward operator or design matrix)
    mprior : array-like
        Vector of the mean of the prior distribution.
    covm : array-like
        Covariance matrix for the prior distribution.
    d : array-like
        The data vector (i.e. observed data).
    covd : array-like
        Data covariance matrix.

    Returns
    -------
    mmap : array-like
        MAP solution.
    covmp : array-like
        Posterior covariance matrix.
    """

    mprior = np.ravel(mprior)
    d = np.ravel(d)

    covm_inv = nla.inv(covm)
    covd_inv = nla.inv(covd)

    covmp = nla.inv(nla.multi_dot([G.T, covd_inv, G]) + covm_inv)

    # Take care of any lack of symmetry
    covmp = (covmp+covmp.T) / 2.0

    # Matrix square roots
    covm_sri = sla.sqrtm(covm_inv, disp=True)
    covd_sri = sla.sqrtm(covd_inv, disp=True)

    A = np.concatenate([np.dot(covd_sri, G), covm_sri], axis=0)
    b = np.concatenate([np.dot(covd_sri, d), np.dot(covm_sri, mprior)])

    m, n = A.shape
    if m == n:
        mmap = nla.solve(A, b)
    else:
        mmap = nla.lstsq(A, b, rcond=None)[0]

    return (mmap, covmp)


def simmvn(mu, cov):
    """
    Generate a random vector that is a realization of an multivariate
    normal (MVN) distribution ``X`` with a known mean, ``mu``, and
    covariance matrix, ``cov``.

    Parameters
    ----------
    mu : array-like
        Vector of expected values.
    cov : array-like
        Covariance matrix.

    Returns
    -------
    result : array-like
        Random vector.
    """

    if np.isscalar(mu):
        mu = np.asarray(mu).reshape(-1,)
    else:
        mu = np.asarray(mu)

        if (mu.ndim > 2) or (mu.ndim == 2 and mu.shape[1] != 1):
            raise ValueError(
                'argument "mu" must be a scalar, 1-D array, or a 2-D vector')

    R = sla.cholesky(cov, lower=False)
    return np.dot(R.T, np.random.randn(mu.size)) + mu


def corrmat(M, corrlen, want_corrfun=False):
    """
    Construct a correlation matrix with columns

        R_{i,.} = shift(a_j, i)

    where ``a_j`` is the desired sequence of parameter correlations,
    with a zero lag correlation of 1. For more details see eq. (11.35)
    in Aster, Borchers and Thurber (2011).

    Parameters
    ----------
    M : int
        Number of rows (and columns) in M-by-M output.

    corrlen : int
        Correlation length (a correlation of ``1/e`` at a lag of
        approximately ``corrlen``).

    want_corrfun : bool, (optional, default: False)
        Whether to return correlation function array in output.

    Returns
    -------
    R : array-like
        ``M-by-M`` array of correlation matrix.

    cfun : array-like (optional)
        Array of correlation function of length ``2M``.
    """

    # Start with a triangular root kernel
    ckern = np.zeros(2*M, dtype=np.float)
    # ckern[M-corrlen-1:M+corrlen] = np.bartlett(2*corrlen + 1)
    ckern[M-corrlen-2:M+corrlen+1] = np.bartlett(2*corrlen + 3)

    # Auto-correlation of root kernel to define a uniform correlation structure
    cfun = acorr(ckern, maxlag=M, normed=True, want_lags=False)

    # Populate the correlation matrix
    R = np.zeros((M, M), dtype=np.float)
    for i in range(M):
        c = np.roll(cfun, i+1+M)
        R[:, i] = c[:M]

    if want_corrfun:
        return R, cfun
    return R
