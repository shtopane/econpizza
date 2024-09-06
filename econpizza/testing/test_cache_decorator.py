import jax.export
import pytest
import jax
import jax.numpy as jnp
from jax import export
from unittest.mock import patch, call
import shutil
import os
import sys
import struct

# autopep8: off
sys.path.insert(0, os.path.abspath("."))
import econpizza as ep
from econpizza.utilities.cache_decorator import cacheable_function_with_export

from econpizza.config import EconPizzaConfig
# autopep8: on

@pytest.fixture(scope="function", autouse=True)
def ep_config_reset():
    ep.config = EconPizzaConfig()

@pytest.fixture(scope="function", autouse=True)
def os_getcwd_create():
      folder_path = "./config_working_dir"

      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

      with patch("os.getcwd", return_value="./config_working_dir"):
        yield
      
      if os.path.exists(folder_path):
        print("remove path", folder_path)
        shutil.rmtree(folder_path)

@pytest.fixture()
def array_of_tuples_and_a_scalar():
    arr1 = jnp.ones(5)
    arr2 = jnp.ones(((2, 3)))
    arr3 = jnp.ones((3, 2))
    scalar = 200

    return (arr1, arr1, arr1), arr2, arr3, scalar
    


@patch("econpizza.utilities.cache_decorator.export_and_serialize")
@patch('builtins.open')
@patch("jax.config.update")
def test_do_not_serialize_when_caching_not_enabled(mock_jax_update, mock_file_write, mock_export_and_serialize):
    @cacheable_function_with_export("f", {
        "x": ("a", jnp.float64)
    })
    def f(x):
        return x**2
    
    x_test = jnp.ones(5)
    res = f(x_test)

    print("in test")
    mock_file_write.assert_not_called()
    mock_export_and_serialize.assert_not_called()


@patch("jax.export.deserialize")
@patch("econpizza.utilities.cache_decorator.export_and_serialize")
@patch("jax.config.update")
def test_export_simple(mock_jax_update, mock_export_and_serialize, mock_deserialize):
    ep.config.update("enable_persistent_cache", True)
    ep.config.update("econpizza_cache_folder", "./config_working_dir")
    
    @cacheable_function_with_export("f", {
        "x": ("a", jnp.float64)
    })
    def f(x):
        return x**2
    
    x_test = jnp.ones(5)
    # On first call, the function is exported
    f(x_test)
    mock_export_and_serialize.assert_called_once()
    # Clear history of calls
    mock_export_and_serialize.reset_mock()
    
    
    f_serialized = bytearray(struct.pack('d'*5, *[1.0]*5))
    with patch("econpizza.utilities.cache_decorator._read_serialized_function", return_value=f_serialized) as mock_read_serialize:
      # On second call, the function is loaded from disk
      f(x_test)
      mock_export_and_serialize.assert_not_called()

@patch('builtins.open')
@patch("jax.export.export")
@patch("jax.config.update")
def test_export_tuple_args(mock_jax_update, mock_export, mock_file_write, array_of_tuples_and_a_scalar):
    ep.config.update("enable_persistent_cache", True)
    ep.config.update("econpizza_cache_folder", "./config_working_dir")
    
    @cacheable_function_with_export("f", {
        "x": (
            (
                ("a", jnp.float64),
                ("a", jnp.float64),
                ("a", jnp.float64)

            ),
            ("b, c", jnp.float64),
            ("c, b", jnp.float64),
            ("", jnp.int64)
        )
    })
    def f1(x):
        return x
    
    f1(array_of_tuples_and_a_scalar)
    mock_export.assert_called_once()

    # Check shapes created from shape_struct
    poly_shape_call = mock_export.mock_calls[1]
    _, poly_args, _ = poly_shape_call
    
    assert len(poly_args[0]) == 4
    assert isinstance(poly_args[0][1], jax.ShapeDtypeStruct)
    assert isinstance(poly_args[0][2], jax.ShapeDtypeStruct)
    assert isinstance(poly_args[0][3], jax.ShapeDtypeStruct)

    assert poly_args[0][1].dtype == jnp.float64
    assert poly_args[0][2].dtype == jnp.float64
    assert poly_args[0][3].dtype == jnp.int64


