"""Configuration object"""

import os
import jax

class EconPizzaConfig(dict):
    def __init__(self, *args, **kwargs):
        super(EconPizzaConfig, self).__init__(*args, **kwargs)
        self._enable_persistent_cache = False
        self._enable_jax_persistent_cache = False

    @property
    def enable_persistent_cache(self):
        return self._enable_persistent_cache

    @enable_persistent_cache.setter
    def enable_persistent_cache(self, value):
        self._enable_persistent_cache = value

    @property
    def enable_jax_persistent_cache(self):
        return self._enable_jax_persistent_cache

    @enable_jax_persistent_cache.setter
    def enable_jax_persistent_cache(self, value):
        self._enable_jax_persistent_cache = value

    def update(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"'EconPizzaConfig' object has no attribute '{key}'")

    def __repr__(self):
        properties = {
            k.lstrip("_"): v for k, v in self.__dict__.items() if k.startswith("_")
        }
        return f"{properties}"

    def __str__(self):
        return self.__repr__()


config = EconPizzaConfig()

def _create_cache_dir(folder_name: str):
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def enable_persistent_cache():
    """Create folders for JAX and EconPizza cache.
    By default, they are created in callee working directory.
    """
    if config.enable_persistent_cache == True:
        _create_cache_dir("econpizza_cache")
        
        folder_name = _create_cache_dir("jax_cache")

        jax.config.update("jax_compilation_cache_dir", folder_name)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

