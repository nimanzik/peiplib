"""
Linear algebra utilities.

Copyright (c) 2017 Nima Nooshiri <nima.nooshiri@gfz-potsdam.de>
"""

import numpy as np
from numpy import linalg as nla
from scipy import linalg as sla


def solve_svd(U, s, V, d, nkeep=None):
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
    Gdagger = nla.multi_dot([Vp, nla.inv(Sp), Up.T])
    return np.dot(Gdagger, d)


def solve_tikh(G, L, alpha, d):
    """
    Return Tikhonov regularization solution (using the SVD or GSVD )

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
    A = np.dot(G.T, G) + alpha*alpha*np.dot(L.T, L)
    Ghash = np.dot(nla.inv(A), G.T)
    return np.dot(Ghash, d)


class MatrixColumnMismatch(Exception):
    pass


def _diagk(X, k):
    """
    K-th matrix diagonal.
    This function returns the k-th diagonal of X except for 1-D arrays.
    """
    X = np.asanyarray(X)
    s = X.shape
    if len(s) > 1:
        D = np.diag(X, k)
    else:
        D = np.array([])

    return D


def _diagf(X):
    """Diagonal force.
    This function zeros all the elements off the main diagonal of X.
    """
    return np.triu(np.tril(X))


def _diagp(Y, X, k):
    """
    Diagonal positive.
    This function scales the columns of Y and the rows of X by unimodular
    factors to make the k-th diagonal of X real and positive.
    """
    D = _diagk(X, k)
    j = np.where((np.real(D) < 0) | (np.imag(D) != 0))[0]
    D = np.diag(np.conj(D[j]) / np.abs(D[j]))
    Y[:, j] = np.dot(Y[:, j], D.T)
    X[j, :] = np.dot(D, X[j, :])

    return (Y, X)


def _csd(Q1, Q2):
    """
    Cosine-Sine Decomposition.

    Given Q1 and Q2 such that Q1'*Q1 + Q2'*Q2 = I, the C-S Decomposition is a
    joint factorization of the form
        Q1 = U*C*Z' and Q2=V*S*Z'
    where U, V, and Z are orthogonal matrices and C and S are diagonal matrices
    (not necessarily square) satisfying
        C'*C + S'*S = I
    """
    m, p = Q1.shape
    n, pb = Q2.shape
    if pb != p:
        raise MatrixColumnMismatch()

    if m < n:
        V, U, Z, S, C = _csd(Q2, Q1)
        j = np.arange(p)[-1::-1]
        C = C[:, j]
        S = S[:, j]
        Z = Z[:, j]
        m = min(m, p)
        i = np.arange(m)[-1::-1]
        C[:m, :] = C[i, :]
        U[:, :m] = U[:, i]
        n = min(n, p)
        i = np.arange(n)[-1::-1]
        S[:n, :] = S[i, :]
        V[:, :n] = V[:, i]
        return (U, V, Z, C, S)

    # Henceforth, (n <= m)
    U, c, Zh = sla.svd(Q1)
    C = sla.diagsvd(c, *Q1.shape)
    Z = np.transpose(Zh)

    q = min(m, p)
    i = np.arange(q)
    j = np.arange(q)[-1::-1]
    C[:q, :q] = C[q-1::-1, q-1::-1]
    U[:, i] = U[:, j]
    Z[:, i] = Z[:, j]
    S = np.dot(Q2, Z)

    if q == 1:
        k = 0
    elif m < p:
        k = n
    else:
        dum = np.nonzero(np.diag(C) <= 1.0 / np.sqrt(2))[0]
        if not np.any(dum):
            k = 0
        else:
            k = max(dum) + 1

    if k == 0:
        V = np.identity(S[:, :k].shape[0])
    else:
        V, _ = sla.qr(S[:, :k])

    S = np.dot(V.T, S)

    r = min(k, m)
    S[:, :r] = _diagf(S[:, :r])
    if (m == 1) and (p > 1):
        S[0, 0] = 0

    if k < min(n, p):
        r = min(n, p)
        i = np.arange(k, n)
        j = np.arange(k, r)
        UT, sT, VhT = sla.svd(S[k:n, k:r])
        ST = sla.diagsvd(sT, n-k, r-k)
        VT = np.transpose(VhT)
        if k > 0:
            S[:k, k:r] = 0

        S[k:n, k:r] = ST
        C[:, j] = np.dot(C[:, j], VT)
        V[:, i] = np.dot(V[:, i], UT)
        Z[:, j] = np.dot(Z[:, j], VT)

        i = np.arange(k, q)
        Q, R = sla.qr(C[k:q, k:r])
        C[k:q, k:r] = _diagf(R)
        U[:, i] = np.dot(U[:, i], Q)

    if m < p:
        # Diagonalize final block of S and permute blocks.
        try:
            eps = np.finfo(C.dtype).eps
        except ValueError:
            eps = np.finfo(np.float).eps

        dum1 = np.count_nonzero(np.abs(_diagk(C, 0)) > 10*m*eps)
        dum2 = np.count_nonzero(np.abs(_diagk(S, 0)) > 10*n*eps)
        q = min(dum1, dum2)
        i = np.arange(q, n)
        j = np.arange(m, p)

        # At this point, S(i,j) should have orthogonal columns and the
        # elements of S(:,q+1:p) outside of S(i,j) should be negligible.
        Q, R = sla.qr(S[q:n, m:p])
        S[:, q:p] = 0
        S[q:n, m:p] = _diagf(R)
        V[:, i] = np.dot(V[:, i], Q)

        if n > 1:
            i = np.hstack([
                np.arange(q, q + p - m),
                np.arange(q),
                np.arange(q + p - m, n)])
        else:
            i = 1

        j = np.hstack([np.arange(m, p), np.arange(m)])
        C = C[:, j]
        S = S[i, :][:, j]
        Z = Z[:, j]
        V = V[:, i]

    if n < p:
        # Final block of S is negligible.
        S[:, n:p] = 0

    # Make sure C and S are real and positive.
    U, C = _diagp(U, C, max(0, p-m))
    C = np.real(C)
    V, S = _diagp(V, S, 0)
    S = np.real(S)

    return (U, V, Z, C, S)


def gsvd(A, B, full_matrices=True, compute_all=True):
    """Generalized Singular Value Decomposition.

    [U,V,X,C,S] = GSVD(A,B) returns unitary matrices U and V, a (usually)
    square matrix X, and nonnegative diagonal matrices C and S so that:

        A = U*C*X'
        B = V*S*X'
        C'*C + S'*S = I

    A and B must have the same number of columns, but may have different
    numbers of rows.  If A is m-by-p and B is n-by-p, then U is m-by-m, V is
    n-by-n and X is p-by-q where q = min(m+n,p).
    """
    m, p = A.shape
    n, pb = B.shape
    if pb != p:
        raise MatrixColumnMismatch()

    QA = np.array([])
    QB = np.array([])
    if not full_matrices:
        # Economy-sized
        if m > p:
            QA, A = sla.qr(A, mode='economic')
            QA, A = _diagp(QA, A, 0)
            m = p

        if n > p:
            QB, B = sla.qr(B, mode='economic')
            QB, B = _diagp(QB, B, 0)
            n = p

    Q, R = sla.qr(np.vstack([A, B]), mode='economic')
    U, V, Z, C, S = _csd(Q[0:m, :], Q[m:m+n, :])

    if compute_all:
        # Full composition.
        X = np.dot(R.T, Z)
        if QA.size != 0:
            U = np.dot(QA, U)

        if QB.size != 0:
            V = np.dot(QB, V)

        return (U, V, X, C, S)

    else:
        # Vector of generalized singular values.
        q = min(m+n, p)
        dum1 = np.zeros((q-m, 1), dtype=np.float)
        dum2 = _diagk(C, max(0, q-m)).reshape(-1, 1)
        dumA = np.vstack([dum1, dum2])

        dum3 = _diagk(S, 0).reshape(-1, 1)
        dum4 = np.zeros((q-n, 1))
        dumB = np.vstack([dum3, dum4])

        sigma = dumA / dumB
        return sigma


__all__ = """
solve_svd
solve_tikh
gsvd
""".split()
