"""interpolation tools
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax._src.typing import Array

import sys

interpolate_numpy = jnp.vectorize(jnp.interp, signature='(nq),(n),(n)->(nq)')

@partial(jnp.vectorize, signature='(n),(nq),(n)->(nq)')
def interpolate(x: Array, xq: Array, y: Array) -> Array: # x (a, ), xq (b, ), y (a, )
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending input points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """

    nx = x.shape[0]

    xi = jnp.minimum(jnp.searchsorted(x, xq, side='right') - 1, nx-2)
    xqpi_cur = (x[xi + 1] - xq) / (x[xi + 1] - x[xi])
    yq = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

    return yq

# TODO: Not used
def interpolate_fast(xp: Array, x: Array, fp: Array) -> Array:
    """
    One-dimensional linear interpolation for monotonically increasing sample points.
    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    The vmap'ed version is much faster. Note: does not extrapolate linearly!
    """
    return jax.vmap(jnp.interp)(jax.lax.broadcast(x, (xp.shape[0],)), xp, fp)


def interpolate_coord_robust_vector(x: Array, xq: Array) -> (Array, Array): # x(a,), xq(b, )
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """

    nx = x.shape[0]

    xqi = jnp.minimum(jnp.searchsorted(x, xq, side='right') - 1, nx-2)
    xqpi = (x[xqi+1] - xq) / (x[xqi+1] - x[xqi])

    return xqi, xqpi


interpolate_coord = jnp.vectorize(
    interpolate_coord_robust_vector, signature='(nq),(nq)->(nq),(nq)')

# xq dimensions change between models - not appropriate for exporting?
def interpolate_coord_robust(x: Array, xq: Array, check_increasing=False) -> (Array, Array):
    """Linear interpolation exploiting monotonicity only in data x, not in query points xq.
    Simple binary search, less efficient but more robust.
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Main application intended to be universally-valid interpolation of policy rules.
    Dimension k is optional.

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (k, nq), query points (in any order)

    Returns
    ----------
    xqi  : array (k, nq), indices of lower bracketing gridpoints
    xqpi : array (k, nq), weights on lower bracketing gridpoints
    """
    if x.ndim != 1:
        raise ValueError(
            'Data input to interpolate_coord_robust must have exactly one dimension')

    if check_increasing and jnp.any(x[:-1] >= x[1:]):
        raise ValueError(
            'Data input to interpolate_coord_robust must be strictly increasing')

    if xq.ndim == 1:
        return interpolate_coord_robust_vector(x, xq)
    else:
        i, pi = interpolate_coord_robust_vector(x, xq.ravel())
        return i.reshape(xq.shape), pi.reshape(xq.shape)


# used in hank2
@partial(jnp.vectorize, signature='(nq),(nq),(n)->(nq)')
def apply_coord(x_i: Array, x_pi: Array, y: Array) -> Array: # x_i: (a, ), x_pi: (a, ), y: (a, )
    """Use representation xqi, xqpi to get yq at xq:
    yq = xqpi * y[xqi] + (1-xqpi) * y[xqi+1]

    Parameters
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """
    return x_pi*y[x_i] + (1-x_pi)*y[x_i+1]

# used in hank 2
@partial(jnp.vectorize, signature='(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs: Array, rhs: Array) -> (Array, Array): # lhs: (a, ), rhs: (a, a)
    print(lhs.shape, rhs.shape)
    # sys.exit()
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

        lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]

    i.e. where given j, lhs == rhs in between i and i+1.

    Also return the pi such that

        pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.
    """

    rhs_check = rhs.at[-1, :].set(jnp.inf)

    ii = jnp.sum(lhs > rhs_check, 0)
    jj = jnp.arange(lhs.shape[0])

    iout = jnp.maximum(ii - 1, 0)
    err_upper = rhs[ii, jj] - lhs[ii]
    err_lower = rhs[ii - 1, jj] - lhs[ii - 1]
    piout = jnp.where(ii == 0, 1, err_upper / (err_upper - err_lower))

    return iout, piout
