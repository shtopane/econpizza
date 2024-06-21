import jax
import jax.numpy as jnp
from jax import export

from jax_aot_export import serialized_f


jax.config.update('jax_compilation_cache_dir', 'docs/tutorial/cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0000000000001)


def f(x, y):
  return 2 * x * y

x, y = 3, 4


jit_f = jax.jit(f)

i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype('int32'))
aot_f = jax.jit(f).lower(i32_scalar, i32_scalar).compile()

# print("AOT answer: ",aot_f(3.0, y))
# print("JIT answer: ",jit_f(3.0, y))

# Hydrate serialized function from another file

rehydrated_exp: export.Exported = export.deserialize(serialized_f)

def callee(y):
  return 3 * rehydrated_exp.call(y * 4)

print(callee(1))
print(type(callee(1)))