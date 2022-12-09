# Lispy: Lisp-like Python magic

```py
import lispy

(Def @Factorial (n)
    (If (Eq @n @0)
        @1
        (Mul @n (Factorial (Sub @n @1)))))

(Print (Factorial @10))  # => 3628800
```

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

For usage in scripts, simply add `import lispy` to the first line of the file

For an interactive REPL, run `python -m lispy`

## Syntax

As a dialect of Lisp, [S-expressions](https://en.wikipedia.org/wiki/S-expression)
are used to denote code. Unfortunately, expressions in the form `(a b c)` are
syntax errors in Python, so it is necessary to add "filler operators" between
expression atoms to convince it otherwise, like so: `(a @b @c)`

Anything that is not a constant, identifier or parenthesis is ignored.

## Built-in functions

TODO
