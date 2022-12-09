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

The [examples](examples/) folder showcases some of the language's features.

