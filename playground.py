from copy import deepcopy
import econpizza as ep

from jax.scipy.linalg import lu_factor
import numpy as np

import pickle
import jax.numpy as jnp

from objexplore import explore
import jax

from jax import export

example_hank = ep.examples.hank
example_hank2 = ep.examples.hank2
example_dsge = ep.examples.dsge

mod = ep.load(example_dsge)
_ = mod.solve_stst()

## Manually store func_eqns for a HANK model. Testing
# shapes_and_dtypes = [
#     jax.ShapeDtypeStruct((18, 1), np.float64),  # XLag
#     jax.ShapeDtypeStruct((18, 1), np.float64),  # X
#     jax.ShapeDtypeStruct((18, 1), np.float64),  # XPrime
#     jax.ShapeDtypeStruct((18, 1), np.float64),  # XSS
#     jax.ShapeDtypeStruct((3,), np.float64),     # shocks
#     jax.ShapeDtypeStruct((10,), np.float64),    # pars
#     jax.ShapeDtypeStruct((1, 4, 50, 1), np.float64),  # distributions
#     jax.ShapeDtypeStruct((2, 4, 50, 1), np.float64)   # decisions_outputs
# ]

# exported_func_eqns: export.Exported = export.export(jax.jit(mod['context']['func_eqns']))(*shapes_and_dtypes)

# print(exported_func_eqns)
# print(exported_func_eqns.fun_name)
# print(exported_func_eqns.in_avals)

## Store manually compiled function equations to disk
# serialized_func_eqns: bytearray = exported_func_eqns.serialize()

# with open('model_func_eqns.pkl', 'wb') as file:
#     pickle.dump(serialized_func_eqns, file)


# print(mod['context'])

## Run model with a lot of shocks
durations = []
shocks = np.linspace(0.001, 0.1, 1)

variables = ['e_beta', 'e_i', 'e_z']
shock_tuples = [(var, 0.004) for var in variables]

for shk_value in shocks:
  x, flat = mod.find_path(shock=('e_z', shk_value))
  # duration = deepcopy(mod['cache']['durations'])
  # durations.append(duration)
