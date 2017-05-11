from itertools import count

import numpy as np
from numpy import linalg as la
from scipy.sparse import csr_matrix


def lcurve_tikh_svd(U, s, d, npoints, alpha_min=None, alpha_max=None):
    '''L-curve parameters for Tikhonov standard-form regularization.

    Parameters
    ----------
    U: ndarray
        Matrix of data space basis vectors from the SVD.
    s: ndarray
        Vector of singular values from the SVD.
    d: ndarray
        The data vector.
    npoints: int
        Number of logarithmically spaced regularization parameters.
    alpha_min: float
        If specified, minimum of the regularization parameters range.
    alpha_max:
        If specified, maximum of the reqularization parameters range.

    Returns
    -------
    rho: ndarray
        Vector of residual norm ``||Gm-d||_2``.
    eta: ndarray
        Vector of solution norm ``||m||_2``.
    reg_params: ndarray
        Vector of corresponding regularization parameters.

    .. seealso:
        Hansen, P. C. (2001), The L-curve and its use in the numerical
        treatment of inverse problems, in book: Computational Inverse Problems
        in Electrocardiology, pp 119-142.
    '''
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

    smin_ratio = 16 * np.finfo(np.float).eps
    start = alpha_min or max(s[-1], s[0]*smin_ratio)
    stop = alpha_max or s[0]
    reg_params = np.linspace(np.log10(start), np.log10(stop), npoints)
    reg_params = 10**reg_params

    s2 = s**2
    for i in range(npoints):
        f = s2 / (s2 + reg_params[i]**2)
        eta[i] = la.norm(f * d_proj_scale)
        rho[i] = la.norm((1-f) * d_proj)

    # If we couldn't match the data exactly add the projection induced misfit
    if (m > n) and (dr > 0):
        rho = np.sqrt(rho**2 + dr)

    return (rho, eta, reg_params)


def lcurve_tikh_gsvd(U, X, LAM, MU, d, G, L, npoints, alpha_min=None,
        alpha_max=None):
    '''
    L-curve parameters for Tikhonov general-form regularization.

    If the system matrix G is m-by-n and the corresponding roughening matrix L
    is p-by-n, then the generalized singular value decomposition (GSVD) of
    A=[G; L] is:

        U, V, X, LAM, MU = gsvd(G, L)

    Parameters
    ----------
    U: ndarray
        m-by-m matrix of data space basis vectors from the GSVD.
    X: ndarray
        n-by-n nonsingular matrix computed by the GSVD.
    LAM: ndarray
        m-by-n matrix, computed by the GSVD, with diagonal entries that may be
        shifted from the main diagonal.
    MU: ndarray
        p-by-n diagonal matrix computed by the GSVD.
    d: ndarray
        The data vector.
    G: ndarray
        The system matrix.
    L: ndarray
        The roughening matrix.
    npoints: int
        Number of logarithmically spaced regularization parameters.
    alpha_min: float
        If specified, minimum of the regularization parameters range.
    alpha_max:
        If specified, maximum of the reqularization parameters range.

    Returns
    -------
    rho: ndarray
        Vector of residual norm ``||Gm-d||_2``.
    eta: ndarray
        Vector of solution seminorm ``||Lm||_2``.
    reg_params: ndarray
        Vector of corresponding regularization parameters.

    .. seealso:
        Aster, R., Borchers, B. & Thurber, C. (2011), `Parameter Estimation and
        Inverse Problems`, Elsevier, pp 103-107.
    '''
    m, n = G.shape
    p = la.matrix_rank(L)

    if len(d.shape) == 2:
        d = d.reshape(d.size,)

    lams = np.sqrt(np.diag(np.dot(LAM.T, LAM)))
    mus = np.sqrt(np.diag(np.dot(MU.T, MU)))
    gammas = lams / mus
    Y = np.transpose(la.inv(X))

    # Initialization
    mod = np.zeros((n, npoints), dtype=np.float)
    eta = np.zeros(npoints, dtype=np.float)
    rho = np.zeros_like(eta)

    if alpha_min and alpha_max:
        start = alpha_min
        stop = alpha_max
    else:
        gmin_ratio = 16 * np.finfo(np.float).eps
        if m <= n:
            # The under-determined or square case.
            k = n - m
            i1, i2 = sorted((k, p-1))
            start = max(gammas[i1], gammas[i2]*gmin_ratio)
            stop = gammas[i2]
        else:
            # The over-determined case.
            start = max(gammas[0], gammas[p-1]*gmin_ratio)
            stop = gammas[p-1]

    reg_params = np.linspace(np.log10(start), np.log10(stop), npoints)
    reg_params = 10**reg_params


    if m > n:
        k = 0
    else:
        k = n - m

    ngam = gammas.size

    # Solve for each solution.
    for ireg in range(npoints):

        # Series filter coeficients for this regularization parameter.
        f = np.zeros(ngam, dtype=np.float)
        for igam in range(ngam):
            gam = gammas[igam]

            if np.isinf(gam) or np.isnan(gam):
                f[igam] = 1
            elif (lams[igam] == 0) and (mus[j] == 0):
                f[igam] = 0
            else:
                f[igam] = gam**2 / (gam**2 + reg_params[ireg]**2)

            # Build the solution (see Aster er al. (2011), eq. (4.49)).
            mod[:, ireg] += f[igam] * (np.dot(U[:, igam-k].T, d)/lams[igam]) * Y[:, igam]

        rho[ireg] = la.norm(np.dot(G, mod[:, ireg]) - d)
        eta[ireg] = la.norm(np.dot(L, mod[:, ireg]))

    return (rho, eta, reg_params)


def lcurve_corner(rho, eta, reg_params):
    '''Triangular/circumscribed circle simple approximation to curvature.

    Parameters
    ----------
    rho: ndarray
        Vector of residual norm ``||Gm-d||_2``.
    eta: ndarray
        Vector of solution norm ``||m||_2`` or seminorm ``||Lm||_2``
    reg_params: ndarray
        Vector of corresponding regularization parameters.

    Returns
    -------
    corner: float
        The value of `reg_params` with maximum curvature.
    icorner: int
        The index of the value in `reg_params` with maximum curvature.
    kappa: ndarray
        Curvature values for each regularization parameter.

    .. seealso:
        https://en.wikipedia.org/wiki/Circumscribed_circle#Triangle_centers_on_the_circumcircle_of_triangle_ABC
    '''
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
    a = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    b = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    c = np.sqrt((x1-x3)**2 + (y1-y3)**2)

    # semiperimeter
    s = (a + b + c)/2

    # circumradii
    R = (a * b * c) / (4 * np.sqrt(s * (s-a) * (s-b) * (s-c)))

    # The curvature for each estimate for each value is the reciprocal of its
    # circumradius. Since there are not circumcircle for end points, their
    # curvature is zero.
    kappa = np.hstack((0, 1.0/R, 0))
    icorner = np.argmax(abs(kappa[1:-1]))
    corner = reg_params[icorner]

    return (corner, icorner, kappa)


def get_reg_mat(n, d, full=True):
    '''1-D differentiating matrix (reffered to as regularization or roughening
    matrix `L`).
    This function computes the discrete approximation `L` to the derivative
    operator of order `d` on a regular grid with `n` points, i.e. L is
    (n-d)-by-n.

    Parameters
    ----------
    n: int
        Number of data points.
    d: int
        The order of the derivative to approximate.
    full: bool
        If True (default), it computes the full matrix. Otherwise it returns a
        sparse matrix.

    Returns
    -------
    L: ndarray or :py:class:``scipy.sparse.csr.csr_matrix``
        The discrete differentiation matrix operator.

    .. seealso:
        https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29
        http://netlib.org/linalg/html_templates/node91.html
    '''

    # Zero'th order derivative
    if d == 0:
        return np.identity(n)

    # Let df approximates the first derivative
    df = np.insert(np.zeros((1, d-1), dtype=np.float), 0, [-1, 1])

    for i in range(1, d):
        # Take the difference of the lower order derivative and itself shifted
        # left to get a derivative one order higher.
        df = np.insert(df[0:d], 0, 0) - np.append(df[0:d], 0)

    nd = n - d
    vals = np.tile(df, nd)
    c = count(start=0, step=(d+1))
    rowptrs = []
    colinds = []
    for i in range(nd):
        rowptrs.append(c.next())
        colinds.extend(range(i, i+d+1))

    # By convension, rowptrs[end]=nnz, where nnz is the number of nonzeros in L
    rowptrs.append(len(vals))

    L = csr_matrix((vals, colinds, rowptrs), shape=[nd, n])

    if full:
        return L.toarray()
    return L
