"""Decorator to handle function serialization"""
import econpizza as ep
import os
from jax import export
import jax
import sys

def export_and_serialize(func, func_name, shape_struct, vjp_order, skip_jitting):
    """
    Export and serialize a function with given symbolic shapes.

    Args:
        func (function): The function to be exported and serialized. If `skip_jitting` is True, then `func` needs to be jitted.
        func_name (str): The name of the function to be used for the serialized file.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int): The order of the vector-Jacobian product.
        skip_jitting (bool): Whether to skip JIT compilation.

    Returns:
        function: The exported and serialized function ready to be called.
    """
    scope = export.SymbolicScope()

    # Verify that each shape and dtype are compatible
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
    
    function_to_export = func if skip_jitting else jax.jit(func)
    exported_func: export.Exported = export.export(function_to_export)(*args)

    serialized_path = os.path.join(ep.config.econpizza_cache_folder, f"{func_name}")
    serialized: bytearray = exported_func.serialize(vjp_order=vjp_order)

    with open(serialized_path, "wb") as file:
      file.write(serialized)

    return exported_func.call


def cacheable_function_with_export(func_name, shape_struct, vjp_order = 0, skip_jitting = False):
    """
    Decorator to replace function with exported and cached version if caching is enabled.

    Args:
        func_name (str): The name under which the function will be saved. "my_func" will be saved as "my_func.bin" on disk.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int, optional): The order of the vector-Jacobian product. Defaults to 0.
        skip_jitting (bool, optional): Whether to skip JIT compilation. Defaults to False. 

    Returns:
        function: The decorated function which uses the cached version if available, otherwise the original function.
    
    Usage:
        @cacheable_function_with_export("f", {"x": ("a,", jnp.float64)})
        def f(x):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):

            if ep.config.enable_persistent_cache == True:
                serialized_path = os.path.join(
                    ep.config.econpizza_cache_folder, f"{func_name}"
                )

                if os.path.exists(serialized_path):
                    with open(serialized_path, "rb") as file:
                        serialized = file.read()
                else:
                    serialized = None

                if serialized:
                    cached_func = export.deserialize(serialized)
                    return cached_func.call(*args)
                else:
                    # Export, serialize, and cache the function
                    cached_func = export_and_serialize(func, func_name, shape_struct, vjp_order, skip_jitting)
                    return cached_func(*args, **kwargs)
            else:
                # Just use the original function
                return func(*args, **kwargs)

        return wrapper

    return decorator
