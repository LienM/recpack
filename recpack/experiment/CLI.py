import argparse
import functools
from collections import defaultdict

import inspect


class CLI(object):

    def __init__(self):
        self.commands = dict()
        self.description = ""
        self.parser = argparse.ArgumentParser(description=self.description)

    @property
    def Command(self):
        class Command(object):

            function_kwargs = dict()

            def __init__(self):
                super().__init__()

            def __init_subclass__(cls, interface=None, **kwargs):
                super().__init_subclass__(**kwargs)
                self.commands[cls.__name__] = (cls, interface)

            @classmethod
            def update_functions(cls, function_kwargs):
                cls.function_kwargs = function_kwargs
                for fName, kwargs in function_kwargs.items():
                    print(fName, kwargs)
                    f = getattr(cls, fName)
                    wrapper = functools.partialmethod(f, **kwargs)
                    setattr(cls, fName, wrapper)

            def get_params(self):
                params = dict()
                for kwargs in self.__class__.function_kwargs.values():
                    params.update(kwargs)
                return params

            def run(self):
                pass

        return Command

    def run(self):
        if len(self.commands) > 0:
            self.setup()
            cls, function_kwargs = self.parse_args()
            cls.update_functions(function_kwargs)

            inst = cls()
            inst.run()

    def setup(self):
        subparsers = self.parser.add_subparsers(help="Select one of the following subcommands:", dest='command',
                                           metavar="subcommand")
        subparsers.required = True

        for name, (cls, interface) in self.commands.items():
            sub_parser = subparsers.add_parser(name, help=cls.__doc__, description=cls.__doc__)
            for fName, f in inspect.getmembers(cls, predicate=inspect.isfunction):
                for param in inspect.signature(f).parameters.values():
                    if param.name == "self":
                        continue

                    # skip non named arguments and variadic parameters
                    if param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                        continue

                    interfaceF = getattr(interface, fName, None)
                    if interfaceF is not None and param.name in [p.name for p in inspect.signature(interfaceF).parameters.values()]:
                        continue

                    tpe = param.annotation
                    if tpe is inspect.Parameter.empty:
                        tpe = str
                    prefix = "-" if len(param.name) == 1 else "--"
                    if param.default is not inspect.Parameter.empty:
                        sub_parser.add_argument(prefix + param.name,
                                                help="type: {}, default={}".format(tpe.__name__, param.default),
                                                type=tpe, default=param.default, dest=f"{fName}.{param.name}")
                    else:
                        sub_parser.add_argument(prefix + param.name, help="type: " + tpe.__name__,
                                                type=tpe, dest=f"{fName}.{param.name}")

    def parse_args(self):
        cmd_args = self.parser.parse_args()
        fName = cmd_args.command
        cls, _ = self.commands[fName]

        kwargs = {n: v for n, v in cmd_args._get_kwargs() if n != "command"}

        function_kwargs = defaultdict(dict)
        for (fName, kwarg), val in ((kwarg.split("."), val) for kwarg, val in kwargs.items()):
            function_kwargs[fName][kwarg] = val

        return cls, function_kwargs
