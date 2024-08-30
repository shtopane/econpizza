"""Decorator to handle function serialization"""
import econpizza as ep
import os
from jax import export
import jax
import sys

def export_and_serialize(func, func_name, shape_struct, vjp_order=0, skip_jitting=False, export_kwargs=False):
    """
    Export and serialize a function with given symbolic shapes.

    Args:
        func (function): The function to be exported and serialized. If `skip_jitting` is True, then `func` needs to be jitted.
        func_name (str): The name of the function to be used for the serialized file.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int): The order of the vector-Jacobian product.
        skip_jitting (bool): Whether to skip JIT compilation.
        export_kwargs (bool): Whether to export the function using keyword arguments instead of positional(default)
    Returns:
        function: The exported and serialized function ready to be called.
    """
    scope = export.SymbolicScope()
    function_to_export = func if skip_jitting else jax.jit(func)
    # Verify that each shape and dtype are compatible
    try:
        poly_args = map_shape_struct_dict_to_jax_shape(shape_struct, scope)
    except ValueError as e:
        print(f"Error in shape or dtype construction: {e}")
        raise

    if export_kwargs == True:
        func_kwargs_names = list(shape_struct.keys())
        poly_kwargs = {key: value for key, value in zip(func_kwargs_names, poly_args)}
        exported_func: export.Exported = export.export(function_to_export)(**poly_kwargs)
    else:
        exported_func: export.Exported = export.export(function_to_export)(*poly_args)
    
    serialized_path = os.path.join(ep.config.econpizza_cache_folder, f"{func_name}")
    serialized: bytearray = exported_func.serialize(vjp_order=vjp_order)

    with open(serialized_path, "wb") as file:
      file.write(serialized)

    return exported_func.call


def cacheable_function_with_export(func_name, shape_struct, vjp_order = 0, skip_jitting = False, export_kwargs=False):
    """
    Decorator to replace function with exported and cached version if caching is enabled.

    Args:
        func_name (str): The name under which the function will be saved. "my_func" will be saved as "my_func.bin" on disk.
        shape_struct (dict): A dictionary defining the shape and type of the function's inputs.
        vjp_order (int, optional): The order of the vector-Jacobian product. Defaults to 0.
        skip_jitting (bool, optional): Whether to skip JIT compilation. Defaults to False. 
        export_kwargs (bool): Whether to export the function using keyword arguments instead of positional(default)

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
                    return cached_func.call(*args, **kwargs)
                else:
                    # Export, serialize, and cache the function
                    cached_func = export_and_serialize(func, func_name, shape_struct, vjp_order, skip_jitting, export_kwargs)
                    return cached_func(*args, **kwargs)
            else:
                # Just use the original function
                return func(*args, **kwargs)

        return wrapper

    return decorator

def map_shape_struct_dict_to_jax_shape(node, scope):
    if isinstance(node, tuple) and len(node) == 2 and not isinstance(node[0], tuple):
        value, dtype = node
        shape_poly = export.symbolic_shape(shape_spec=value, scope=scope)
        return jax.ShapeDtypeStruct(shape_poly, dtype=dtype)
    elif isinstance(node, dict):
        return tuple(map_shape_struct_dict_to_jax_shape(v, scope) for v in node.values())
    elif isinstance(node, (list, tuple)):
        return type(node)(map_shape_struct_dict_to_jax_shape(v, scope) for v in node)
    else:
        return node

