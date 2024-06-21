from jax import export
import jax
import jax.numpy as jnp

def f(x): return 2 * x * x

# aot_f = jax.jit(f).lower(jax.ShapeDtypeStruct((), jnp.int32)).compile()

exported: export.Exported = export.export(jax.jit(f))(jax.ShapeDtypeStruct((), jnp.int32))

serialized_f: bytearray = exported.serialize()