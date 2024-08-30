"""Internal subfunctions for heterogeneous agent models
"""

import jax
import jax.numpy as jnp
from jax._src.typing import Array
from typing import Callable

from econpizza.utilities.export.cache_decorator import cacheable_function_with_export

# TODO:[caching] this as well?

# @see backwards_sweep_stst
def _backwards_stst_cond(carry):
    _, (wf, _), (wf_old, cnt), (_, tol, maxit) = carry
    cond0 = jnp.abs(wf - wf_old).max() > tol
    cond1 = cnt < maxit
    return jnp.logical_and(cond0, cond1)

# @see backwards_sweep_stst
# TODO: [function]
def _backwards_stst_body(carry):
    (x, par), (wf, _), (_, cnt), (func, tol, maxit) = carry
    return (x, par), func(x, x, x, x, wf, pars=par), (wf, cnt + 1), (func, tol, maxit)

# x (a, 1)
# par (b, )
# carry (tuple [3])
#  carry[0][0] (1, 4, 50) (HANK) (2, 3, 10, 20) (HANK2)
#  carry[0][1] (2, 4, 50) (HANK) (4, 3, 10, 20) (HANK2)
#  carry[1][0] (1, 4, 50) (HANK) (2, 3, 10, 20) (HANK2)
#  carry[1][1] int64() (HANK, HANK2)
#  carry[2][0] Partial func_backw
#  carry[2][1] float64 (HANK) (HANK2)
# TODO: [function]
def backwards_sweep_stst(x, par, carry):
    _, (wf, decisions_output), (_, cnt), _ = jax.lax.while_loop(
        _backwards_stst_cond, _backwards_stst_body, ((x, par), *carry))
    return wf, decisions_output, cnt

# see backwards_sweep
# TODO: [function]
def _backwards_step(carry, i):

    wf, X, shocks, func_backw, stst, pars = carry
    wf, decisions_output = func_backw(
        X[:, i], X[:, i+1], X[:, i+2], WFPrime=wf, shocks=shocks[:, i], pars=pars)

    return (wf, X, shocks, func_backw, stst, pars), (wf, decisions_output)

# x (3582,) (a * b)
# x0 (a, )
# shocks (c, b)
# pars (d, )
# stst (a, )
# wfSS (1, 4, 50) (e, f, g )
# @cacheable_function_with_export("backwards_sweep", {
#     "x": ("a*b", jnp.float64),
#     "x0": ("a", jnp.float64),
#     "shocks": ("c, b", jnp.float64),
#     "pars": ("d", jnp.int64),
#     "stst": ("a", jnp.float64),
#     "wfSS": ("e, f, g", jnp.float64)
# })
def backwards_sweep(x: Array, x0: Array, shocks: Array, pars: Array, stst: Array, wfSS: Array, horizon: int, func_backw: Callable, return_wf=False) -> Array:

    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T

    _, (wf_storage, decisions_output_storage) = jax.lax.scan(
        _backwards_step, (wfSS, X, shocks, func_backw, stst, pars), jnp.arange(horizon-1), reverse=True)
    decisions_output_storage = jnp.moveaxis(decisions_output_storage, 0, -1)
    wf_storage = jnp.moveaxis(wf_storage, 0, -1)

    if return_wf:
        return wf_storage, decisions_output_storage
    return decisions_output_storage

# @see forward_sweep
# TODO: [function]
def _forwards_step(carry, i):

    dist_old, decisions_output_storage, func_forw = carry
    dist = func_forw(dist_old, decisions_output_storage[..., i])

    return (dist, decisions_output_storage, func_forw), dist_old

# decisions_output_storage (2, 4, 50, 199) (a, f, g, b) (example poly shape)
# dist0 (1, 4, 50) (e, f, g ) (example poly)
# TODO: [function]
def forwards_sweep(decisions_output_storage: Array, dist0: Array, horizon: int, func_forw: callable) -> Array:

    _, dists_storage = jax.lax.scan(
        _forwards_step, (dist0, decisions_output_storage, func_forw), jnp.arange(horizon-1))
    dists_storage = jnp.moveaxis(dists_storage, 0, -1)

    return dists_storage

# x (5572) (a * b)
# dists_storage (1, 3, 10, 20, 199) (HANK2) (1, e, f, g, b) (example shape poly)
# decisions_output_storage (4, 3, 10, 20, 199) (HANK2) (4, e, f, g, b) (example shape poly)
# x0 (28, ) (a, )
# shocks (5, 199) (c, b)
# pars (20, ) (d, )
# stst (28, ) (a, )
# TODO: [function]
def final_step(x: Array, dists_storage: Array, decisions_output_storage: Array, x0: Array, shocks: Array, pars: Array, stst: Array, horizon: int, nshpe, func_eqns: Callable) -> Array:

    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T
    out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
        nshpe), stst, shocks, pars, dists_storage, decisions_output_storage)

    return out

# TODO: [function]
# x (5572) (a * b)
# decisions_output_storage (4, 3, 10, 20, 199) (HANK2) (4, e, f, g, b) (example shape poly)
# x0 (28, ) (a, )
# dist0 (1, 3, 10, 20) (HANK2) (1, e, f, g) (example shape poly)
# shocks (5, 199) (c, b)
# pars (20, ) (d, )
def second_sweep(x: Array, decisions_output_storage: Array, x0: Array, dist0: Array, shocks: Array, pars: Array, forwards_sweep: Callable, final_step: Callable) -> Array:

    # forwards step
    dists_storage = forwards_sweep(decisions_output_storage, dist0)
    # final step
    out = final_step(x, dists_storage,
                     decisions_output_storage, x0, shocks, pars)

    return out

# TODO: [function]
# x (5572) (a * b)
# x0 (28, ) (a, )
# dist0 (1, 3, 10, 20) (HANK2) (1, e, f, g) (example shape poly)
# shocks (5, 199) (c, b)
# pars (20, ) (d, )
def stacked_func_het_agents(x: Array, x0: Array, dist0: Array, shocks: Array, pars: Array, backwards_sweep: Callable, second_sweep: Callable):

    # backwards step
    decisions_output_storage = backwards_sweep(x, x0, shocks, pars)
    # combined step
    out = second_sweep(x, decisions_output_storage, x0, dist0, shocks, pars)

    return out
