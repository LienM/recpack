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

            def __init_subclass__(cls, mask=None, **kwargs):
                super().__init_subclass__(**kwargs)
                self.commands[cls.__name__] = (cls, mask)

            @classmethod
            def update_functions(cls, function_kwargs):
                cls.function_kwargs = function_kwargs
                for fName, kwargs in function_kwargs.items():
                    f = getattr(cls, fName)
                    wrapper = functools.partialmethod(f, **kwargs)
                    setattr(cls, fName, wrapper)

            def get_params(self):
                params = super().get_params() if hasattr(super(), "get_params") else dict()
                for kwargs in self.__class__.function_kwargs.values():
                    params.update(kwargs.items())
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
        subparsers = self.parser.add_subparsers(
            help="Select one of the following subcommands:",
            dest='command',
            metavar="subcommand"
        )
        subparsers.required = True

        for name, (cls, mask) in self.commands.items():
            sub_parser = subparsers.add_parser(
                name,
                help=cls.__doc__,
                description=cls.__doc__,
                formatter_class=argparse.MetavarTypeHelpFormatter
            )
            for fName, f in inspect.getmembers(cls, predicate=inspect.isfunction):
                for param in inspect.signature(f).parameters.values():
                    if param.name == "self":
                        continue

                    # skip non named arguments and variadic parameters
                    if param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                        continue

                    # skip parameters that are part of mask
                    maskF = getattr(mask, fName, None)
                    if maskF is not None and param.name in [p.name for p in inspect.signature(maskF).parameters.values()]:
                        continue

                    param_name = param.name[1:] if param.name[0] == "_" else param.name
                    prefix = "-" if len(param_name) == 1 else "--"
                    if param.default is not inspect.Parameter.empty:
                        tpe = param.annotation
                        if tpe is inspect.Parameter.empty:
                            tpe = type(param.default)

                        if tpe == bool:
                            group = sub_parser.add_mutually_exclusive_group(required=False)
                            group.add_argument('--' + param_name, help="(default)" if param.default else "", dest=f"{fName}.{param.name}", action='store_true')
                            group.add_argument('--no-' + param_name, help="(default)" if not param.default else "", dest=f"{fName}.{param.name}", action='store_false')
                            group.set_defaults(**{f"{fName}.{param.name}": param.default})
                        else:
                            sub_parser.add_argument(prefix + param_name,
                                                    help="(default: {})".format(param.default),
                                                    type=tpe, default=param.default, dest=f"{fName}.{param.name}")
                    else:
                        tpe = param.annotation
                        if tpe is inspect.Parameter.empty:
                            tpe = str

                        if tpe == bool:
                            group = sub_parser.add_mutually_exclusive_group(required=True)
                            group.add_argument('--' + param_name, dest=f"{fName}.{param.name}", action='store_true')
                            group.add_argument('--no-' + param_name, dest=f"{fName}.{param.name}", action='store_false')
                        else:
                            sub_parser.add_argument(prefix + param_name, type=tpe, dest=f"{fName}.{param.name}")

    def parse_args(self):
        cmd_args = self.parser.parse_args()
        fName = cmd_args.command
        cls, _ = self.commands[fName]

        kwargs = {n: v for n, v in cmd_args._get_kwargs() if n != "command"}

        function_kwargs = defaultdict(dict)
        for (fName, kwarg), val in ((kwarg.split("."), val) for kwarg, val in kwargs.items()):
            function_kwargs[fName][kwarg] = val

        return cls, function_kwargs
