"""Configuration object"""

import os
import jax

class EconPizzaConfig(dict):
    def __init__(self, *args, **kwargs):
        super(EconPizzaConfig, self).__init__(*args, **kwargs)
        self._enable_persistent_cache = False
        self._cache_folder_pizza = None
        self._cache_folder_jax = None

    @property
    def enable_persistent_cache(self):
        return self._enable_persistent_cache

    @enable_persistent_cache.setter
    def enable_persistent_cache(self, value):
        self._enable_persistent_cache = value

    @property
    def cache_folder_jax(self):
        return self._cache_folder_jax
    
    @cache_folder_jax.setter
    def cache_folder_jax(self, value):
        self._cache_folder_jax = value
    
    @property
    def cache_folder_pizza(self):
        return self._cache_folder_pizza
    
    @cache_folder_pizza.setter
    def cache_folder_pizza(self, value):
        self._cache_folder_pizza = value

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
        folder_path_pizza = _create_cache_dir("__econpizza_cache__")
        folder_path_jax = _create_cache_dir("__jax_cache__")

        jax.config.update("jax_compilation_cache_dir", folder_path_jax)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        config.cache_folder_jax = folder_path_jax
        config.cache_folder_pizza = folder_path_pizza

