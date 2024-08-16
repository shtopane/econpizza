"""Tools for dealing with distributions
"""

import jax
import jax.numpy as jnp
from jax._src.typing import Array
 # Pi[3,3], X[3, 10, 20]
def expect_transition(Pi: Array, X: Array) -> Array:
    """If Pi is a matrix, multiply Pi times the ith dimension of X and return"""
    shape = X.shape
    # iterate forward using Pi
    X = Pi @ X.reshape((shape[0], -1))
    # reverse steps
    X = X.reshape((Pi.shape[0], *shape[1:]))

    return X

 # D[4, 50], x_i[4, 50], x_pi[4, 50]
def forward_policy_1d(D: Array, x_i: Array, x_pi: Array) -> Array:
    nZ, _ = D.shape
    Dnew = jnp.zeros_like(D)

    j = jnp.arange(nZ)[:, None]

    Dnew = Dnew.at[j, x_i].add(D * x_pi)
    Dnew = Dnew.at[j, x_i+1].add(D * (1 - x_pi))

    return Dnew


def cond_func(carry):
    (dist, cnt, dist_old), (tol, maxit), _ = carry
    cond0 = jnp.abs(dist-dist_old).max() > tol
    cond1 = cnt < maxit
    return jnp.logical_and(cond0, cond1)


def _body_func_1d(carry):
    (dist, cnt, _), cond_vars, exo_endo = carry
    exog_probs, endog_inds, endog_probs = exo_endo
    dist_new = exog_probs.T @ forward_policy_1d(dist, endog_inds, endog_probs)
    return (dist_new, cnt + 1, dist), cond_vars, exo_endo

# endog_inds[4, 50], endog_probs[4, 50], exog_probs: [4,4]
def stationary_distribution_forward_policy_1d(endog_inds: Array, endog_probs: Array, exog_probs: Array, tol=1e-10, maxit=1000) -> (Array, int):
    dist = jnp.ones_like(endog_inds, dtype=jnp.float64)
    dist /= dist.sum()

    (dist, cnt, _), _, _ = jax.lax.while_loop(
        cond_func, _body_func_1d, ((dist, 0, dist+1), (tol, maxit), (exog_probs, endog_inds, endog_probs)))
    return dist, cnt


# D[3, 10, 20], x_i(3, 10, 20) y_i[3, 10, 20], x_pi(3, 10, 20) y_pi[3, 10, 20]
def forward_policy_2d(D: Array, x_i: Array, y_i: Array, x_pi: Array, y_pi: Array) -> Array:

    nZ, _, _ = D.shape
    Dnew = jnp.zeros_like(D)

    j = jnp.arange(nZ)[:, None, None]

    Dnew = Dnew.at[j, x_i, y_i].add(y_pi * x_pi * D)
    Dnew = Dnew.at[j, x_i+1, y_i].add(y_pi * (1 - x_pi) * D)
    Dnew = Dnew.at[j, x_i, y_i+1].add((1 - y_pi) * x_pi * D)
    Dnew = Dnew.at[j, x_i+1, y_i+1].add((1 - y_pi) * (1 - x_pi) * D)

    return Dnew


def _body_func_2d(carry):
    (dist, cnt, _), cond_vars, exo_endo = carry
    exog_probs, endog_inds0, endog_inds1, endog_probs0, endog_probs1 = exo_endo
    pre_exo_dist = forward_policy_2d(
        dist, endog_inds0, endog_inds1, endog_probs0, endog_probs1)
    new_dist = expect_transition(exog_probs.T, pre_exo_dist)
    return (new_dist, cnt + 1, dist), cond_vars, exo_endo

# endog_inds0(3, 10, 20) endog_inds1(3, 10, 20), endog_probs0(3, 10, 20), endog_probs1(3, 10, 20), exog_probs(3, 3)
def stationary_distribution_forward_policy_2d(endog_inds0: Array, endog_inds1: Array, endog_probs0: Array, endog_probs1: Array, exog_probs: Array, tol=1e-10, maxit=1000) -> (Array, int):
    # TODO: can be merged with stationary_distribution_forward_policy_1d

    dist = jnp.ones_like(endog_inds0, dtype=jnp.float64)
    dist /= dist.sum()

    exo_endo = exog_probs, endog_inds0, endog_inds1, endog_probs0, endog_probs1
    (dist, cnt, _), _, _ = jax.lax.while_loop(cond_func,
                                              _body_func_2d, ((dist, 0, dist+1), (tol, maxit), exo_endo))

    return dist, cnt
 # T[4,4](hank1), T[3,3](hank2) -> T[a, a]
def stationary_distribution(T):
    """Find invariant distribution of a Markov chain by unit eigenvector.
    NOTE: jax has no autodiff support for eig. (there is a version with custom_jvp in grgrwip)
    """
    v, w = jnp.linalg.eig(T)

    # using sorted args instead of np.isclose is neccessary for jax-jitting
    args = jnp.argsort(v)
    unit_ev = w[:, args[-1]]

    return unit_ev.real / unit_ev.real.sum()
