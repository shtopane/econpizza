"""Decorator to handle function serialization"""

import econpizza as ep
import os

from jax import export

import jax.numpy as jnp
import jax
import sys

def export_and_serialize(
    args,
    kwargs,
    func,
    func_name,
    shape_struct,
    vjp_order,
    skip_jitting,
    export_with_kwargs=False,
    reuse_first_item_scope=False,
):
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

    if reuse_first_item_scope == True:
        poly_args = _prepare_poly_args_with_first_item(shape_struct)
    else:
        scope = export.SymbolicScope()
        poly_args = map_shape_struct_dict_to_jax_shape(shape_struct, scope)

    function_to_export = func if skip_jitting else jax.jit(func)

    if export_with_kwargs == True:
        poly_kwargs = _prepare_poly_kwargs(shape_struct, kwargs, poly_args)

        exported_func: export.Exported = export.export(function_to_export)(
            **poly_kwargs
        )
    else:
        exported_func: export.Exported = export.export(function_to_export)(*poly_args)

    serialized_path = os.path.join(ep.config.econpizza_cache_folder, f"{func_name}")
    serialized: bytearray = exported_func.serialize(vjp_order=vjp_order)

    # Save the serialized object to the serialized path
    with open(serialized_path, "wb") as file:
        file.write(serialized)

    return exported_func.call


def cacheable_function_with_export(
    func_name,
    shape_struct,
    vjp_order=0,
    skip_jitting=False,
    export_with_kwargs=False,
    reuse_first_item_scope=False,
):
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
      @cacheable_function_with_export("f", {"x": ("a,", jnp.float64)}
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if ep.config.enable_persistent_cache == True:

                serialized_path = os.path.join(
                    ep.config.econpizza_cache_folder, f"{func_name}.bin"
                )
                serialized = _read_serialized_function(serialized_path)

                # kwargs should only be the ones in shape_struct, if len(kwargs) > len(shape_struct)
                # only arguments from shape_struct should be left out
                filtered_kwargs = {
                    key: value for key, value in kwargs.items() if key in shape_struct
                }

                if serialized:
                    cached_func = export.deserialize(serialized)
                    return cached_func.call(*args, **filtered_kwargs)
                else:
                    # Export, serialize, and cache the function
                    cached_func = export_and_serialize(
                        args,
                        kwargs,
                        func,
                        func_name,
                        shape_struct,
                        vjp_order,
                        skip_jitting,
                        export_with_kwargs,
                        reuse_first_item_scope,
                    )
                    return cached_func(*args, **filtered_kwargs)
            else:
                # Just use the original function
                return func(*args, **kwargs)

        return wrapper

    return decorator


def map_shape_struct_dict_to_jax_shape(node, scope):
    """Generate a jax.ShapeDTypeStruct from a dictionary of polymorphic shapes.

    Args:
        node (tuple | dict | list): an element from the shape (example: ("a", jnp.int64))
        scope (export.SymbolicScope): the scope which the symbolic shape should use. As this is constructing a whole
        structure with related objects, they should share the same scope

    Returns:
        jax.ShapeDtypeStruct: shape struct using symbolic shape
    """
    if (
        isinstance(node, tuple)
        and (len(node) == 2 or len(node) == 3)
        and not isinstance(node[0], tuple)
    ):
        if len(node) == 3:
            value, dtype, constraint = node
        else:
            value, dtype = node
            constraint = None

        if constraint:
            shape_poly = export.symbolic_shape(shape_spec=value, constraints=constraint)
        else:
            shape_poly = export.symbolic_shape(shape_spec=value, scope=scope)

        if scope is None:
            return jax.ShapeDtypeStruct(shape_poly, dtype=dtype), shape_poly[0].scope
        else:
            return jax.ShapeDtypeStruct(shape_poly, dtype=dtype)
    elif isinstance(node, dict):
        return tuple(
            map_shape_struct_dict_to_jax_shape(v, scope) for v in node.values()
        )
    elif isinstance(node, (list, tuple)):
        return type(node)(map_shape_struct_dict_to_jax_shape(v, scope) for v in node)
    else:
        return node


def _read_serialized_function(serialized_path):
    if os.path.exists(serialized_path):
        with open(serialized_path, "rb") as file:
            serialized = file.read()
    else:
        serialized = None

    return serialized


def _prepare_poly_args_with_first_item(shape_struct):
    first_item_key, first_item_value = next(iter(shape_struct.items()))
    first_item_poly_args, first_item_scope = map_shape_struct_dict_to_jax_shape(
        first_item_value, scope=None
    )

    rest_items = {
        key: value for key, value in shape_struct.items() if key != first_item_key
    }
    rest_item_poly_args = map_shape_struct_dict_to_jax_shape(
        rest_items, scope=first_item_scope
    )

    poly_args = (first_item_poly_args,) + rest_item_poly_args

    assert len(poly_args) == len(
        shape_struct
    ), "Shape poly arguments are not the same as originally provided in shape_struct"
    return poly_args


def _prepare_poly_kwargs(shape_struct, kwargs, poly_args):
    func_kwargs_names = list(shape_struct.keys())
    poly_kwargs = {key: value for key, value in zip(func_kwargs_names, poly_args)}
    # poly_kwargs.update(kwargs)

    for key, value in kwargs.items():
        if key not in poly_kwargs:
            poly_kwargs[key] = value

    assert len(poly_kwargs) == len(kwargs), "Keyword argument missing in shape poly"

    return poly_kwargs
