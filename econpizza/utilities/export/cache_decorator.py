"""Decorator to handle function serialization"""
import econpizza as ep
import os
from jax import export
import jax
import sys

def export_and_serialize(func, func_name, shape_struct, vjp_order):
    """Export and serialize a function with given symbolic shapes."""
    scope = export.SymbolicScope()

    # Verify that each shape and dtype is compatible
    try:
        args = [
            jax.ShapeDtypeStruct(
                export.symbolic_shape(shape, scope=scope), dtype=dtype
            )
            for shape, dtype in shape_struct.values()
        ]
    except ValueError as e:
        print(f"Error in shape or dtype construction: {e}")
        raise

    exported_func: export.Exported = export.export(jax.jit(func))(*args)
    # Serialize the exported function
    serialized_path = os.path.join(ep.config.cache_folder_pizza, f"{func_name}.bin")
    # os.makedirs(serialized_path, exist_ok=True)
   
    serialized: bytearray = exported_func.serialize(vjp_order=vjp_order)

    # Save the serialized object to the serialized path
    with open(serialized_path, "wb") as file:
      file.write(serialized)

    return exported_func.call


def cacheable_function_with_export(func_name, shape_struct, vjp_order = 0):
    """Decorator to replace function with exported and cached version if caching is enabled.
    Usage:
      @cacheable_function_with_export("f", {"x": ("a,", jnp.float64)}
    """

    def decorator(func):
        def wrapper(*args, **kwargs):

            if ep.config.enable_persistent_cache:

                serialized_path = os.path.join(
                    ep.config.cache_folder_pizza, f"{func_name}.bin"
                )

                if os.path.exists(serialized_path):
                    # Load the cached function
                    with open(serialized_path, 'rb') as file:
                        serialized = file.read()
                else:
                    serialized = None

                if serialized:
                    cached_func = export.deserialize(serialized)
                    return cached_func.call(*args)
                else:
                    # Export, serialize, and cache the function
                    cached_func = export_and_serialize(func, func_name, shape_struct, vjp_order)
                    return cached_func(*args)
            else:
                # Just use the original function
                return func(*args, **kwargs)

        return wrapper

    return decorator
