"""Exporting functions from `interp.py` module"""

from jax import export
import jax
import jax.numpy as jnp

from econpizza.utilities.interp import apply_coord, interpolate


def export_apply_coord():

  scope = export.SymbolicScope()

  shape = export.symbolic_shape(
      "a, ", scope=scope
  )
  # xq_symbolic_shape = export.symbolic_shape(
  # "b, ", scope=scope
  # )

  shape_struct = {
  "x_i": jax.ShapeDtypeStruct(shape, dtype=jnp.int64),
  "x_pi": jax.ShapeDtypeStruct(shape, dtype=jnp.float64),
  "y": jax.ShapeDtypeStruct(shape, dtype=jnp.float64),
  }

  args = list(shape_struct.values())
  
  interpolate_exported: export.Exported = export.export(jax.jit(apply_coord))(*args)
  print(interpolate_exported.in_avals)