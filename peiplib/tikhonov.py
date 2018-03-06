"""
Tikhonov Regularization tools.

Copyright (c) 2017 Nima Nooshiri <nima.nooshiri@gfz-potsdam.de>
"""

from itertools import count

import numpy as np
from numpy import linalg as la
from scipy.sparse import csr_matrix

from .util import loglinspace


def lcurve_svd(U, s, d, npoints, reg_min=None, reg_max=None):
    """
    L-curve parameters for Tikhonov standard-form regularization.

    If the system matrix ``G`` is m-by-n, then singular value
    decomposition (SVD) of ``G`` is:

        U, s, V = svd(G)

    Parameters
    ----------
    U : array_like
        Matrix of data space basis vectors from the SVD.
    s : array_like
        Vector of singular values from the SVD.
    d : array_like
        The data vector.
    npoints : int
        Number of logarithmically spaced regularization parameters.
    reg_min : float (optional)
        If specified, minimum of the regularization parameters range.
    reg_max : float (optional)
        If specified, maximum of the reqularization parameters range.

    Returns
    -------
    rho : array_like
        Vector of residual norm `||Gm-d||_2`.
    eta : array_like
        Vector of solution norm `||m||_2`.
    reg_params : array_like
        Vector of corresponding regularization parameters.

    References
    ----------
    .. [1] Hansen, P. C. (2001), The L-curve and its use in the
       numerical treatment of inverse problems, in book: Computational
       Inverse Problems in Electrocardiology, pp 119-142.
    """

    smin_ratio = 16 * np.finfo(np.float).eps
    start = reg_max or s[0]
    stop = reg_min or max(s[-1], s[0]*smin_ratio)
    reg_params = loglinspace(start, stop, npoints)

    m, n = U.shape
    p = s.size

    if len(d.shape) == 2:
        d = d.reshape(d.size,)

    # Projection, and residual error introduced by the projection
    d_proj = np.dot(U.T, d)
    dr = la.norm(d)**2 - la.norm(d_proj)**2

    d_proj = d_proj[0:p]

    # Scale series terms by singular values
    d_proj_scale = d_proj / s

    # Initialize storage space
    eta = np.zeros(npoints, dtype=np.float)
    rho = np.zeros_like(eta)

    s2 = s**2
    for i in range(npoints):
        f = s2 / (s2 + reg_params[i]**2)
        eta[i] = la.norm(f * d_proj_scale)
        rho[i] = la.norm((1-f) * d_proj)

    # If we couldn't match the data exactly add the projection induced misfit
    if (m > n) and (dr > 0):
        rho = np.sqrt(rho**2 + dr)

    return (rho, eta, reg_params)


def lcurve_gsvd(
        U, X, LAM, MU, d, G, L, npoints, reg_min=None, reg_max=None):
    """
    L-curve parameters for Tikhonov general-form regularization.

    If the system matrix ``G`` is m-by-n and the corresponding
    roughening matrix ``L`` is p-by-n, then the generalized singular
    value decomposition (GSVD) of ``A=[G; L]`` is:

        U, V, X, LAM, MU = gsvd(G, L)

    Parameters
    ----------
    U : array_like
        m-by-m matrix of data space basis vectors from the GSVD.
    X : array_like
        n-by-n nonsingular matrix computed by the GSVD.
    LAM : array_like
        m-by-n matrix, computed by the GSVD, with diagonal entries that
        may be shifted from the main diagonal.
    MU : array_like
        p-by-n diagonal matrix computed by the GSVD.
    d : array_like
        The data vector.
    G : array_like
        The system matrix (forward operator or design matrix).
    L : array_like
        The roughening matrix.
    npoints : int
        Number of logarithmically spaced regularization parameters.
    reg_min : float (optional)
        Minimum of the regularization parameters range.
    reg_max : float (optional)
        Maximum of the reqularization parameters range.

    Returns
    -------
    rho : array_like
        Vector of residual norm `||Gm-d||_2`.
    eta : array_like
        Vector of solution seminorm `||Lm||_2`.
    reg_params : array_like
        Vector of corresponding regularization parameters.

    References
    ----------
    .. [1] Aster, R., Borchers, B. & Thurber, C. (2011), `Parameter
       Estimation and Inverse Problems`, Elsevier, pp 103-107.
    """

    m, n = G.shape
    p = la.matrix_rank(L)

    if len(d.shape) == 2:
        d = d.reshape(d.size,)

    lams = np.sqrt(np.diag(np.dot(LAM.T, LAM)))
    mus = np.sqrt(np.diag(np.dot(MU.T, MU)))
    gammas = lams / mus

    if reg_min and reg_max:
        start = reg_max
        stop = reg_min
    else:
        gmin_ratio = 16 * np.finfo(np.float).eps
        if m <= n:
            # The under-determined or square case.
            k = n - m
            i1, i2 = sorted((k, p-1))
            start = reg_max or gammas[i2]
            stop = reg_min or max(gammas[i1], gammas[i2]*gmin_ratio)
        else:
            # The over-determined case.
            start = reg_max or gammas[p-1]
            stop = reg_min or max(gammas[0], gammas[p-1]*gmin_ratio)

    reg_params = loglinspace(start, stop, npoints)

    if m > n:
        k = 0
    else:
        k = n - m

    # Initialization.
    eta = np.zeros(npoints, dtype=np.float)
    rho = np.zeros_like(eta)

    # Solve for each solution.
    for ireg in range(npoints):

        # Series filter coeficients for this regularization parameter.
        f = np.zeros(n, dtype=np.float)
        for igam in range(k, n):
            gam = gammas[igam]

            if np.isinf(gam) or np.isnan(gam):
                f[igam] = 1
            elif (lams[igam] == 0) and (mus[igam] == 0):
                f[igam] = 0
            else:
                f[igam] = gam**2 / (gam**2 + reg_params[ireg]**2)

        # Build the solution (see Aster er al. (2011), eq. (4.49) & (4.56)).
        d_proj_scale = np.dot(U[:, :n-k].T, d) / lams[k:n]
        Y = np.transpose(la.inv(X))
        F = np.diag(f)
        mod = np.dot(Y[:, k:], np.dot(F, d_proj_scale))
        rho[ireg] = la.norm(np.dot(G, mod) - d)
        eta[ireg] = la.norm(np.dot(L, mod))

    return (rho, eta, reg_params)


def lcorner_kappa(rho, eta, reg_params):
    """
    Determination of Tikhonov regularization parameter using L-curve criterion.

    Triangular/circumscribed circle simple approximation to curvature.

    Parameters
    ----------
    rho : array_like
        Vector of residual norm `||Gm-d||_2`.
    eta : array_like
        Vector of solution norm `||m||_2` or seminorm `||Lm||_2`.
    reg_params : array_like
        Vector of corresponding regularization parameters.

    Returns
    -------
    reg_c : float
        The value of regularization parameter corresponding to the
        corner of the L-curve (i.e. the value of ``reg_params`` with
        maximum curvature).
    rho_c : float
        The residual norm corresponding to ``reg_c``.
    eta_c : float
        The solution norm/seminorm corresponding to ``reg_c``.

    References
    ----------
    .. [1] https://de.mathworks.com/matlabcentral/answers/284245-matlab-code-for-computing-curvature-equation#answer_222173
    .. [2] https://en.wikipedia.org/wiki/Circumscribed_circle
        #Triangle_centers_on_the_circumcircle_of_triangle_ABC
    """

    xs = np.log10(rho)
    ys = np.log10(eta)

    # Side lengths for each triangle
    x1 = xs[0:-2]
    x2 = xs[1:-1]
    x3 = xs[2:]
    y1 = ys[0:-2]
    y2 = ys[1:-1]
    y3 = ys[2:]

    # The side length for each triangle
    a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    b = np.sqrt((x2-x3)**2 + (y2-y3)**2)
    c = np.sqrt((x3-x1)**2 + (y3-y1)**2)

    # Area of triangles
    A = 0.5 * abs((x1-x2)*(y3-y2) - (y1-y2)*(x3-x2))

    # Curvature of circumscribing circle
    kappa = (4.0*A) / (a*b*c)

    icorner = np.nanargmax(kappa)
    reg_c = reg_params[icorner]
    rho_c = rho[icorner]
    eta_c = eta[icorner]

    return (reg_c, rho_c, eta_c)


def lcorner_mdf_svd(U, s, d, reg_init=None, tol=1.0e-16, maxiter=1200):
    """
    Determination of Tikhonov regularization parameter using L-curve criterion.

    Minimum distance function (MDF) optimization.

    Parameters
    ----------
    U : array_like
        Matrix of data space basis vectors from the SVD.
    s : array_like
        Vector of singular values from the SVD.
    d : array_like
        The data vector.
    reg_init : float (optional)
        An appropriate initial regularization parameter.
    tol : float
        Absolute error in ``reg_c`` between iterations that is
        acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.

    Returns
    -------
    reg_c : float
        The value of regularization parameter corresponding to the
        corner of the L-curve.
    rho_c : float
        The residual norm corresponding to ``reg_c``.
    eta_c : float
        The solution norm/seminorm corresponding to ``reg_c``.

    References
    ----------
    .. [1] Belge, M., Kilmer, M. E. & Miller, E. L. (2002), `Efficient
        determination of multiple regularization parameters in a
        generalized L-curve framework`, Inverse Problems, 18, 1161-1183.
    """

    # Origin point O=(a,b)
    rho, eta, reg_params = lcurve_svd(U, s, d, 2)
    a = np.log10(rho[np.argmin(reg_params)]**2)
    b = np.log10(eta[np.argmax(reg_params)]**2)

    if not reg_init:
        q = loglinspace(reg_params[0], reg_params[1], 3)
        reg_init = q[1]

    def f(reg_pre):
        rho_pre, eta_pre, _ = lcurve_svd(
            U, s, d, 1, reg_min=reg_pre, reg_max=reg_pre)
        rho_pre = np.asscalar(rho_pre)
        eta_pre = np.asscalar(eta_pre)
        dum1 = (rho_pre/eta_pre)**2
        dum2 = np.log10(eta_pre**2) - b
        dum3 = np.log10(rho_pre**2) - a
        reg_next = np.sqrt(dum1 * (dum2/dum3))
        return reg_next

    reg_next = f(reg_init)
    change = abs((reg_next/reg_init) - 1.0)

    counter = 1
    while (change > tol) and (counter < maxiter):
        reg_pre = reg_next
        reg_next = f(reg_pre)
        change = abs((reg_next/reg_pre) - 1.0)
        counter += 1

    rho_c, eta_c, reg_c = map(
        np.asscalar,
        lcurve_svd(U, s, d, 1, reg_min=reg_next, reg_max=reg_next))

    return (reg_c, rho_c, eta_c)


def lcorner_mdf_gsvd(
        U, X, LAM, MU, d, G, L, reg_init=None, tol=1.0e-16, maxiter=1200):
    """
    Determination of Tikhonov regularization parameter using L-curve criterion.

    Minimum distance function (MDF) optimization.

    Parameters
    ----------
    U : array_like
        m-by-m matrix of data space basis vectors from the GSVD.
    X : array_like
        n-by-n nonsingular matrix computed by the GSVD.
    LAM : array_like
        m-by-n matrix, computed by the GSVD, with diagonal entries that
        may be shifted from the main diagonal.
    MU : array_like
        p-by-n diagonal matrix computed by the GSVD.
    d : array_like
        The data vector.
    G : array_like
        The system matrix (forward operator or design matrix).
    L : array_like
        The roughening matrix.
    reg_init : float (optional)
        An appropriate initial regularization parameter.
    tol : float
        Absolute error in ``reg_c`` between iterations that is
        acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.

    Returns
    -------
    reg_c : float
        The value of regularization parameter corresponding to the
        corner of the L-curve.
    rho_c : float
        The residual norm corresponding to ``reg_c``.
    eta_c : float
        The solution norm/seminorm corresponding to ``reg_c``.

    References
    ----------
    .. [1] Belge, M., Kilmer, M. E. & Miller, E. L. (2002), `Efficient
        determination of multiple regularization parameters in a
        generalized L-curve framework`, Inverse Problems, 18, 1161-1183.
    """

    # Origin point O=(a,b)
    rho, eta, reg_params = lcurve_gsvd(U, X, LAM, MU, d, G, L, 2)
    a = np.log10(rho[np.argmin(reg_params)]**2)
    b = np.log10(eta[np.argmax(reg_params)]**2)

    if not reg_init:
        q = loglinspace(reg_params[0], reg_params[1], 9)
        reg_init = q[4]

    def f(reg_pre):
        rho_pre, eta_pre, _ = lcurve_gsvd(
            U, X, LAM, MU, d, G, L, 1, reg_min=reg_pre, reg_max=reg_pre)
        rho_pre = np.asscalar(rho_pre)
        eta_pre = np.asscalar(eta_pre)
        dum1 = (rho_pre/eta_pre)**2
        dum2 = np.log10(eta_pre**2) - b
        dum3 = np.log10(rho_pre**2) - a
        reg_next = np.sqrt(dum1 * (dum2/dum3))
        return reg_next

    reg_next = f(reg_init)
    change = abs((reg_next/reg_init) - 1.0)

    counter = 1
    while (change > tol) and (counter < maxiter):
        reg_pre = reg_next
        reg_next = f(reg_pre)
        change = abs((reg_next/reg_pre) - 1.0)
        counter += 1

    rho_c, eta_c, reg_c = map(np.asscalar, lcurve_gsvd(
        U, X, LAM, MU, d, G, L, 1, reg_min=reg_next, reg_max=reg_next))

    return (reg_c, rho_c, eta_c)


def get_rough_mat(n, order, full=True):
    """
    1-D differentiating matrix (reffered to as regularization or
    roughening matrix ``L``).

    This function computes the discrete approximation ``L`` to the
    derivative operator of order ``order`` on a regular grid with ``n``
    points, i.e. ``L`` is ``(n - order)-by-n``.

    Parameters
    ----------
    n : int
        Number of data points.
    order : int
        The order of the derivative to approximate.
    full : bool
        If True (default), it computes the full matrix. Otherwise it
        returns a sparse matrix.

    Returns
    -------
    L : array_like or :py:class:`scipy.sparse.csr.csr_matrix`
        The discrete differentiation matrix operator.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sparse_matrix
        #Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29
    .. [2] http://netlib.org/linalg/html_templates/node91.html
    """

    # Zero'th order derivative.
    if order == 0:
        return np.identity(n)

    # Let df approximates the first derivative.
    df = np.insert(np.zeros((1, order-1), dtype=np.float), 0, [-1, 1])

    for i in range(1, order):
        # Take the difference of the lower order derivative and
        # itself shifted left to get a derivative one order higher.
        df = np.insert(df[0:order], 0, 0) - np.append(df[0:order], 0)

    nd = n - order
    vals = np.tile(df, nd)
    c = count(start=0, step=(order+1))
    rowptrs = []
    colinds = []
    for i in range(nd):
        rowptrs.append(c.next())
        colinds.extend(range(i, i+order+1))

    # By convension, rowptrs[end]=nnz, where nnz is
    # the number of nonzero values in L.
    rowptrs.append(len(vals))

    L = csr_matrix((vals, colinds, rowptrs), shape=[nd, n])

    if full:
        return L.toarray()
    return L
