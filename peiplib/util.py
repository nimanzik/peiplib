"""
Copyright (c) 2021 Nima Nooshiri (@nimanzik)
"""

import math

import numpy as np


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


def shaw_problem(m, n):
    """
    This function generates and returns outgoing angles vector `gamma`
    (eq. 1-40), incoming angles vactor `theta` (eq. 1-41), and forward
    operator G (eq. 1-44) for Shaw problem.

    Parameters
    ----------
    m : int
        Number of data points. `m` must be even.
    n : int
        Number of model points. `n` must be even.

    Returns
    -------
    gamma : array-like
        Data vector, `m` equally spaced outgoing angles.
    theta : array-like
        Model vector, `n` equally spaced incoming angles.
    G : array-like
        Operator G, a 2-D array with shape `(m, n)`.
    """

    i = np.arange(1, m + 1)
    gamma = (i - 0.5) * math.pi / m - math.pi / 2.0

    j = np.arange(1, n + 1)
    theta = (j - 0.5) * math.pi / n - math.pi / 2.0

    dtheta = math.pi / n
    G = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            gamma_i = gamma[i]
            theta_j = theta[j]
            dum1 = math.cos(gamma_i) + math.cos(theta_j)
            dum2 = math.pi * (math.sin(gamma_i) + math.sin(theta_j))
            try:
                G[i, j] = dum1**2 * (math.sin(dum2)/dum2)**2 * dtheta
            except ZeroDivisionError:
                G[i, j] = dum1**2 * 1.0 * dtheta

    return (gamma, theta, G)


__all__ = """
    loglinspace
    nextpow2
    isvector
    shaw_problem
""".split()
