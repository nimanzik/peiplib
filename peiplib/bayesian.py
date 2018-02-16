import numpy as np
from numpy import linalg as nla
from scipy import linalg as sla


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
    mu = np.asarray(mu)
    if (mu.ndim > 2) or (mu.ndim == 2 and mu.shape[1] != 1):
        raise ValueError('array "{}" must be 1-D or 2-D.'.format(mu))

    R = sla.cholesky(cov, lower=False)
    return np.dot(R.T, np.random.randn(mu.size)) + mu
