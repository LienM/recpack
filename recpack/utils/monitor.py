import sys
from tqdm import tqdm


class Monitor(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.tqdm = tqdm(*args,
                         desc=name,
                         leave=True,
                         unit="",
                         maxinterval=1,
                         bar_format="{desc}{bar}[{elapsed}{postfix}]",
                         **kwargs)

    def update(self, description):
        print('\n' * self.tqdm.pos, file=sys.stderr)
        self.tqdm.update()
        self.tqdm.set_description(f"{self.name} - {description}")


