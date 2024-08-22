"""Configuration object"""

import os
import jax

class EconPizzaConfig(dict):
    def __init__(self, *args, **kwargs):
        super(EconPizzaConfig, self).__init__(*args, **kwargs)
        self._enable_persistent_cache = False
        self._econpizza_cache_folder = '__econpizza_cache__'
        self._jax_cache_folder = '__jax_cache__'

    @property
    def enable_persistent_cache(self):
        return self._enable_persistent_cache

    @enable_persistent_cache.setter
    def enable_persistent_cache(self, value):
        self._enable_persistent_cache = value

    @property
    def jax_cache_folder(self):
        return self._jax_cache_folder
    
    @jax_cache_folder.setter
    def jax_cache_folder(self, value):
        self._jax_cache_folder = value
    
    @property
    def econpizza_cache_folder(self):
        return self._econpizza_cache_folder
    
    @econpizza_cache_folder.setter
    def econpizza_cache_folder(self, value):
        self._econpizza_cache_folder = value

    def update(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"'EconPizzaConfig' object has no attribute '{key}'")
    
    def _create_cache_dir(self, folder_name: str):
        cwd = os.getcwd()
        folder_path = os.path.join(cwd, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        return folder_path

    def setup_persistent_cache(self):
        """Create folders for JAX and EconPizza cache.
        By default, they are created in callee working directory.
        """
        if self.enable_persistent_cache == True:
            folder_path_pizza = self._create_cache_dir(self.econpizza_cache_folder)
            folder_path_jax = self._create_cache_dir(self.jax_cache_folder)

            jax.config.update('jax_compilation_cache_dir', folder_path_jax)
            jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
            jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)

            self.jax_cache_folder = folder_path_jax
            self.econpizza_cache_folder = folder_path_pizza

    def __repr__(self):
        properties = {
            k.lstrip("_"): v for k, v in self.__dict__.items() if k.startswith("_")
        }
        return f"{properties}"

    def __str__(self):
        return self.__repr__()


config = EconPizzaConfig()
