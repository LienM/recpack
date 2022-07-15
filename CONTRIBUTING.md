# Code Conventions

Code should loosely follow [pep8](https://www.python.org/dev/peps/pep-0008/).
We use [`black`](https://github.com/ambv/black) for code
formatting and [`flake8`](http://flake8.pycqa.org/en/latest/) for linting its
Python modules.

### Documentation
Documentation is auto generated from docstrings in the code. 
These docstrings should be written in [reStructureText](https://docutils.sourceforge.io/rst.html) format.

For more information on documentation see the [doc/README.md](doc/README.md)

### Code formatting

[`black`](https://github.com/ambv/black) is an opinionated Python code
formatter, optimised to produce readable code and small diffs. You can run
`black` from the command-line, or via your code editor. For example, if you're
using [Visual Studio Code](https://code.visualstudio.com/), you can add the
following to your `settings.json` to use `black` for formatting and auto-format
your files on save:

```json
{
    "python.formatting.provider": "black",
    "[python]": {
        "editor.formatOnSave": true
    }
}
```

[See here](https://github.com/ambv/black#editor-integration) for the full
list of available editor integrations.

### Code linting

[`flake8`](http://flake8.pycqa.org/en/latest/) is a tool for enforcing code
style. It scans one or more files and outputs errors and warnings. This feedback
can help you stick to general standards and conventions, and can be very useful
for spotting potential mistakes and inconsistencies in your code. The most
important things to watch out for are syntax errors and undefined names, but you
also want to keep an eye on unused declared variables or repeated
(i.e. overwritten) dictionary keys. If your code was formatted with `black`
(see above), you shouldn't see any formatting-related warnings.

#### Disabling linting

Sometimes, you explicitly want to write code that's not compatible with our
rules. For example, a module's `__init__.py` might import a function so other
modules can import it from there, but `flake8` will complain about an unused
import. And although it's generally discouraged, there might be cases where it
makes sense to use a bare `except`.

To ignore a given line, you can add a comment like `# noqa: F401`, specifying
the code of the error or warning we want to ignore. It's also possible to
ignore several comma-separated codes at once, e.g. `# noqa: E731,E123`. Here
are some examples:

```python
# The imported class isn't used in this file, but imported here, so it can be
# imported *from* here by another module.
from .submodule import SomeClass  # noqa: F401

try:
    do_something()
except:  # noqa: E722
    # This bare except is justified, for some specific reason
    do_something_else()
```

### Python conventions

Recpack abides by the Zen of Python ([PEP 20 -- The Zen of Python | Python.org](https://www.python.org/dev/peps/pep-0020/)). 

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

Beautiful to us means:
- Clearly defined, well-documented interfaces
- Clear and concise class and method naming. Clean package structure, hierarchy only when needed.
- Code reuse is important, f.e. make util modules or define a method at the highest level in the class hierarchy. 
- Code locality is too, i.e. keep code together that belongs together.  

A few rules in the Zen of Python we wish to highlight:
*Sparse is better than dense.*

Sparsity is inherent to the type of data we're dealing with. 
We prefer using a single-threaded sparse implementation over a parallellized dense implementation that has to deal with a lot of zero rows.

*Readability counts.*

Code is for humans first, then computers. 
If you need to use advanced Python principes for performance reasons, you should clearly document what is going on, so that others after you will be able to understand. 
Usually, simple code does the trick just fine though. 

*Errors should never pass silently.*

It's better to cry wolf when there is none, than to let the wolves feast. 
Same goes for errors. 
Errors should be thrown as soon as possible, preferably before the most intensive computations are done. 

Additionally, we strongly prefer simple inheritance, and will only result to multiple inheritance when we *really* need to. 

Interfaces, interfaces, interfaces! 
Most users are not interested in your code, but in your interfaces. 
Your interfaces are therefore the most important part of your code. 
Define them with care, and document them excessively. 
New interfaces can be defined freely, but when you're re-implementing functionality that is already present in a popular library (numpy, scipy, pandas, scikit-learn), try to maintain consistency. 
This can mean maintaining the same order of method arguments, the same names for kwargs, the same names for methods that do the same things, etc.

We shan't add dependencies if we don't absolutely need them. Extra dependencies are extra opportunities to break stuff.