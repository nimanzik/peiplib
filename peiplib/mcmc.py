"""
Markov Chain Monte Carlo

Copyright (c) 2021 Nima Nooshiri (@nimanzik)
"""

import numpy as np

from peiplib.util import isvector


def metropolis_hastings(
        mstart, nsample, loglikelihood, logprior, proprnd, logprop=None,
        symmetric=False, burnin=0, thin=1, nchain=1):
    """
    Metropolis-Hastings sampling.

    Parameters
    ----------
    mstart : scalar, array-like
        A scalar, 1-D array, or 2-D *vector* containing the value of the
        Markov Chain (i.e. the start model).

    nsmaple : int
        Number of posterior samples to be generated

    loglikelihood : callable
        Function handle providing natural log probability density of the
        likelihood function, i.e. target distribution.

    logprior : callable
        Function handle that computes the natural log probability
        density of the prior distribution.

    proprnd : callable
        Function handle that generates a random (i.e. candidate) model
        from the current model using the proposal distribution.

    logprop : callable
        Function handle providing natural log probability density of the
        proposal distribution. Optional if ``symmetric`` is ``True``, i.e
        the proposal distribution is symmetric.

    symmetric : bool (optional, default: False)
        Whether the proposal distribution is symmetric.

    burnin : int (optional, default: 0)
        Non-negative integer specifying the number of over early samples
        in the chain to be discarded to give time for the algorithm to
        *burn in*.

    thin : int (optional, default: 1)
        Positive integer specifying the decimation factor used to
        down-sample the sequence to produce a low-correlation set of
        posterior samples, i.e. keeping every ``thin``-th draw of the
        chain.

    nchain : int (optional, default: 1)
        Generates ``nchain`` Markov Chains. ``nchain`` is positive integer.

    Returns
    -------
    samples : array-like
        1-D array, if a single chain is generated, or 2-D array, if
        multiple chains are generated, containing the MCMC samples.

    accrate : scalar, array-like
        The acceptance rate of the proposed distribution. ``accrate`` is
        a scalar if a single chain is generated, and is a 1-D array if
        multiple chains are generated.

    mmap : scalar, array-like
        Best model found in the MCMC simulation, i.e. Maximum A
        Posteriori (MAP) solution. It has the same type and size as
        ``mstart``.

    Notes
    -----
    ``loglikelihood``, ``logprior``, and ``proprnd`` take one argumen
    as an input and this argument has the same type and size as
    ``mstart``.

    ``logprop`` takes two arguments as inputs and both arguments have the
    same type and size as ``mstart``.
    """

    if np.isscalar(mstart):
        mstart = np.asarray(mstart, dtype=np.float).reshape(-1,)
        nmod = 1
    else:
        mstart = np.asarray(mstart, dtype=np.float)
        nmod = mstart.size
        if not isvector(mstart):
            raise ValueError(
                'argument "mstart" must be a scalar, 1-D array, or '
                'a 2-D vector')

    if not burnin >= 0:
        raise ValueError('"burnin" must be a non-negative integer')

    if not thin >= 1:
        raise ValueError('"thin" must be positive and >= 1')

    if not nchain >= 1:
        raise ValueError('"nchain" must be positive and >= 1')

    # Allocate space for the results
    samples = np.zeros((nsample, nchain, nmod), dtype=np.float)
    accrate = np.zeros(nchain, dtype=np.float)

    # Place holder for current model
    mj = mstart

    # Uniformly distributed random variables, i.e. t ~ Uniform([0, 1])
    logt = np.log(np.random.rand(nchain, nsample*thin + burnin))

    # Place holders for MAP solution
    mmap = mj
    logmmap = -np.inf

    # Metropolis-Hastings algorithm
    for i in range(-burnin, nsample*thin):
        # Candidate model from proposal dist'n
        mc = proprnd(mj)

        lp_mc = logprior(mc)
        ll_mc = loglikelihood(mc)

        lp_mj = logprior(mj)
        ll_mj = loglikelihood(mj)

        if not symmetric:
            q1 = lp_mc + ll_mc + logprop(mj, mc)
            q2 = lp_mj + ll_mj + logprop(mc, mj)
        else:
            # Save the evaluation time for symmetric proposal dist'n
            q1 = lp_mc + ll_mc
            q2 = lp_mj + ll_mj

        # Evaluation of log(alpha) instead of alpha
        logalpha = min(0., q1-q2)

        # Accept or reject the candidate model
        logt_i = logt[:, i+burnin]
        logt_i = np.log(np.random.rand())

        if logt_i <= logalpha:
            mj = mc
            accrate += 1

            # Update the MAP solution, if this one is better
            x = lp_mc + ll_mc
            if x > logmmap:
                mmap = mj
                logmmap = x

        if i >= 0 and np.mod(i, thin) == 0:
            samples[i/thin, :, :] = mj

    # Acceptance rate can be used to optimize the choice of step size in
    # random-walk Metropolis-Hastings sampler
    accrate /= (nsample*thin + burnin)

    # Rearrange the samples to make it easier to manipulate.
    samples = np.transpose(samples, (0, 2, 1))

    if nchain == 1:
        samples = samples.squeeze(axis=2)

    return (samples, accrate, mmap)


def randomwalk_metropolis_hastings(
        mstart, nsample, loglikelihood, logprior, step,
        proprnd_type='normal', burnin=0, thin=1, nchain=1):
    """
    Random Walk Metropolis-Hastings sampling.

    This is a specific case of the Metropolis-Hastings algorithm, where
    the candidate models are constructed as

        X_{k+1} = X_{k} + e_{k}

    where the e_{k} are chosen to be i.i.d (independent and identically
    distributed) with a symmetric distribution, e.g. Normal or Uniform
    distributions.

    Parameters
    ----------
    mstart : scalar, array-like
        A scalar, 1-D array, or 2-D *vector* containing the value of the
        Markov Chain (i.e. the start model).

    nsmaple : int
        Number of posterior samples to be generated

    loglikelihood : callable
        Function handle providing natural log probability density of the
        likelihood function, i.e. target distribution.

    logprior : callable
        Function handle that computes the natural log probability
        density of the prior distribution.

    step : float
        Positive number specifying the random walk step size (i.e. the
        proposal variance).

    proprnd_type : {'normal', 'uniform'} (optional, default: 'normal')
        Type of random walk proposal:
        if set to 'normal', then X_{k+1} = X_{k} + Normal(0, step**2)
        if set to 'uniform', then X_{k+1} = X_{k} + Uniform([-step, step])

    burnin : int (optional, default: 0)
        Non-negative integer specifying the number of over early samples
        in the chain to be discarded to give time for the algorithm to
        *burn in*.

    thin : int (optional, default: 1)
        Positive integer specifying the decimation factor used to
        down-sample the sequence to produce a low-correlation set of
        posterior samples, i.e. keeping every ``thin``-th draw of the
        chain.

    nchain : int (optional, default: 1)
        Generates ``nchain`` Markov Chains. ``nchain`` is positive integer.

    Returns
    -------
    samples : array-like
        1-D array, if a single chain is generated, or 2-D array, if
        multiple chains are generated, containing the MCMC samples.

    accrate : scalar, array-like
        The acceptance rate of the proposed distribution. ``accrate`` is
        a scalar if a single chain is generated, and is a 1-D array if
        multiple chains are generated.

    mmap : scalar, array-like
        Best model found in the MCMC simulation, i.e. Maximum A
        Posteriori (MAP) solution. It has the same type and size as
        ``mstart``.

    Notes
    -----
    ``loglikelihood``, ``logprior``, take one argumen as an input and
    this argument has the same type and size as ``mstart``.
    """

    if proprnd_type == 'normal':
        def proprnd(x):
            return x + step*np.random.randn(x.size)
    elif proprnd_type == 'uniform':
        def proprnd(x):
            return x + np.random.rand(x.size)*2*step - step
    else:
        raise ValueError(
            'cannot support proprnd_type: "{}"'.format(proprnd_type))

    return metropolis_hastings(
        mstart, nsample, loglikelihood, logprior, proprnd, symmetric=True,
        burnin=burnin, thin=thin, nchain=nchain)


def eval_logpdf(x):
    """
    Computes the natural logarithm of a custom probability density
    function evaluated at ``x``.

    Parameters
    ----------
    x : array-like
        Array of quantiles.

    Returns
    -------
    y : array-like
        Log of the probability density function evaluated at ``x``.
    """
    y = np.full(x.size, -np.inf)
    idx = x > 0
    y[idx] = np.log(x[idx])
    return y


__all__ = """
    metropolis_hastings
    randomwalk_metropolis_hastings
    eval_logpdf
""".split()
