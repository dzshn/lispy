# Lispy: Lisp-like Python magic

```py
import lispy

(Def @Factorial (n)
    (If (Eq @n @0)
        @1
        (Mul @n (Factorial (Sub @n @1)))))

(Print (Factorial @10))  # => 3628800
```

## Features

- Virtually infinite recursion (based on [trampolines](https://en.wikipedia.org/wiki/Trampoline_(computing)))
- Python data types, lisp structure
- Built-in REPL

## TODO

- [x] Trampoline executor
- [ ] Tail recursion optimisation
- [x] More interop with Python
- [ ] Docstrings
- [ ] Do all (or a large subset of) ANSI CL or Scheme or Clojure or another major dialect

## Install

Install with pip: (only requires python >= 3.10)

```sh
$ pip install git+https://github.com/dzshn/lispy
# or `py -m pip` etc
```

From source using [poetry](https://python-poetry.org):
```sh
$ poetry install
```

## Usage

There are two main ways to use the language:

### A. Importing it

When `lispy` is imported, the rest of the file is automatically executed as lisp.

```py
import lispy

(Print@ "hi from lispy :3")
```

Unfortunately, S-expressions are usually illegal syntax in python, so it is necessary
to add "junk operators" to convince it otherwise. Anything that is not a name, constant
or parenthesis is ignored.

## B. Using the CLI

Alternatively, you may run a lispy script as a separate file. Since this doesn't require
Python to parse it as Python code before lispy executes, junk operators are not needed
(although you may still use them)

```sh
# run a script
$ lispy script.lpy
...
# start an interactive REPL
$ lispy
LisPy 0.1.0 on CPython 3.10.8 on Linux
Copyright (c) 2022 Sofia Lima <me@dzshn.xyz>
(>>>)
```

Also check out to the [documentation!](docs/)

## why

[it happens](https://wetdry.world/@z/109482233010304203)
