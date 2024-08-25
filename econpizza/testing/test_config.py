import pytest
import jax
from unittest.mock import patch
import shutil
import os
import sys

sys.path.insert(0, os.path.abspath("."))
import econpizza as ep
from econpizza.config import EconPizzaConfig

@pytest.fixture(scope="function", autouse=True)
def ep_config_reset():
    ep.config = EconPizzaConfig()

@pytest.fixture(scope="function", autouse=True)
def os_getcwd_create():
      folder_path = "./config_working_dir"

      if not os.path.exists(folder_path):
          os.makedirs(folder_path)
          print(f"Created folder: {folder_path}")

      with patch("os.getcwd", return_value="./config_working_dir"):
        yield
      
      if os.path.exists(folder_path):
        print("Deleting folder:", folder_path)
        shutil.rmtree(folder_path)

def test_config_default_values():
  assert ep.config.enable_persistent_cache == False
  assert ep.config.econpizza_cache_folder == "__econpizza_cache__"
  assert ep.config.jax_cache_folder == "__jax_cache__"

def test_config_jax_default_values():
   assert jax.config.values["jax_compilation_cache_dir"] is None
   assert jax.config.values["jax_persistent_cache_min_entry_size_bytes"] == .0
   assert jax.config.values["jax_persistent_cache_min_compile_time_secs"] == 1.0

def test_config_enable_persistent_cache():
   with patch("os.makedirs") as mock_makedirs, patch("jax.config.update") as mock_jax_update:
        ep.config.enable_persistent_cache = True
        mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__econpizza_cache__"), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__jax_cache__"), exist_ok=True)

        mock_jax_update.assert_any_call("jax_compilation_cache_dir", os.path.join(os.getcwd(), "__jax_cache__"))
        mock_jax_update.assert_any_call("jax_persistent_cache_min_entry_size_bytes", -1)
        mock_jax_update.assert_any_call("jax_persistent_cache_min_compile_time_secs", 0)

def test_config_set_econpizza_folder():
   with patch("os.makedirs") as mock_makedirs, patch("jax.config.update") as mock_jax_update:
        ep.config.econpizza_cache_folder = "test1"
        ep.config.enable_persistent_cache = True

        mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "test1"), exist_ok=True)
        mock_jax_update.assert_any_call("jax_compilation_cache_dir", os.path.join(os.getcwd(), "__jax_cache__"))

def test_config_set_jax_folder():
   with patch("os.makedirs") as mock_makedirs, patch("jax.config.update") as mock_jax_update:
        ep.config.jax_cache_folder = "test1"
        ep.config.enable_persistent_cache = True
        mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "test1"), exist_ok=True)
        mock_jax_update.assert_any_call("jax_compilation_cache_dir", os.path.join(os.getcwd(), "test1"))

def test_config_jax_folder_set_from_outside():
    with patch("jax.config.update") as mock_jax_update:
      mock_jax_update("jax_compilation_cache_dir", "jax_from_outside")
      ep.config.enable_persistent_cache = True
      mock_jax_update.assert_any_call("jax_compilation_cache_dir", "jax_from_outside")


def test_econpizza_cache_folder_not_created_second_time():
    ep.config.enable_persistent_cache = True
    assert os.path.exists(ep.config.econpizza_cache_folder)

    with patch("os.makedirs") as mock_makedirs:
        ep.config.enable_persistent_cache = True
        assert mock_makedirs.call_count == 0

def test_config_enable_persistent_cache_called_after_load():
    mod = ep.load(ep.examples.dsge)
    ep.config.enable_persistent_cache = True
    assert os.path.exists(ep.config.econpizza_cache_folder)
    assert os.path.exists(ep.config.jax_cache_folder)


        

