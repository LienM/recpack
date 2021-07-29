import recpack.algorithms
import recpack.metrics


class Registry:
    def __init__(self, src):
        self.registered = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except AttributeError:
            return False

    def get(self, name: str) -> type:
        if name in self.registered:
            return self.registered[name]
        else:
            return getattr(self.src, name)

    def register(self, name: str, c: type):
        self.registered[name] = c


class AlgorithmRegistry(Registry):
    def __init__(self):
        super().__init__(recpack.algorithms)


class MetricRegistry(Registry):
    def __init__(self):
        super().__init__(recpack.metrics)
