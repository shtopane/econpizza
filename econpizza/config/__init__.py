"""Configuration object"""
class EconPizzaConfig(dict):
    def __init__(self, *args, **kwargs):
        super(EconPizzaConfig, self).__init__(*args, **kwargs)
        self._enable_persistent_cache = None
        self._enable_jax_persistent_cache = None

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

config.enable_persistent_cache = False
