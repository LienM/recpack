from typing import Dict

import recpack.algorithms
import recpack.metrics


class Registry:
    def __init__(self, src):
        self.registered: Dict[str, type] = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        """Retrieve the type for the given key.

        :param key: the key of the type to fetch
        :type key: str
        :returns: The class type associated with the key
        :rtype: type
        """
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if the given key is known to the registry.

        :param key: The key to check.
        :type key: str
        :return: True if the key is known
        :rtype: bool
        """
        try:
            self.get(key)
            return True
        except AttributeError:
            return False

    def get(self, key: str) -> type:
        """Retrieve the type for the given key.

        :param key: The key to fetch
        :type key: str
        :return: The class type associated with the key
        :rtype: type
        """
        if key in self.registered:
            return self.registered[key]
        else:
            return getattr(self.src, key)

    def register(self, key: str, c: type):
        """Register a new class.

        After registration, the key can be used to fetch the class from the registry.

        :param key: key to register the type at. Needs to be unique to the registry.
        :type key: str
        :param c: class to register.
        :type c: type
        """
        if key in self:
            raise KeyError(f"key {key} already registered")
        self.registered[key] = c


class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry comes preregistered with all recpack algorithms.
    """

    def __init__(self):
        super().__init__(recpack.algorithms)


class MetricRegistry(Registry):
    """Registry for easy retrieval of metric types by name.

    The registry comes preregistered with all the recpack metrics.
    """

    def __init__(self):
        super().__init__(recpack.metrics)
