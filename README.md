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
- [ ] More interop with Python

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

## Built-ins

### `Def`

Define a function or variable on current scope

```py
(Def@ Square (x)
    (Mod@ x // x))

(Square@ 7)  # => 49

(Def@ Explod (x (y 2))  # y defaults to 0
    (Pow x y))

(Explod@ 9)   # => 81
(Explod@ 3 4) # => 81

(Def@ Pi@ 3)

Pi  # => 3
```

### `Let`

Define variables and execute an expression with a new scope

### `Lambda`

Create and return an anonymous function. same syntax as def (without the function name)

### `Recall`

Call the innermost function, useful for recursion on anonymous functions

```py
((Lambda () (Recall)))  # loop indefinitely (more realistically until it runs out of memory (i really need tco))
```

### `Apply`

Call a function with a given sequence as arguments (roughly `f(*x)`)

### `If`

Conditional function. Execute the second argument if the first is true, otherwise execute the third (if it is present)

### `And`, `Or`
Lazily evaluated logical functions. `And` will execute and return the second argument if the first is true, whereas `Or` will only do so if the first argument is false.

```py
(Def@ Clamp (x)
    (Int (And@ x (Div@ x (Abs@ x)))))

(Clamp@ 143)  # =>  1
(Clamp@ 0)    # =>  0
(Clamp@ -13)  # => -1
```

### `Include`

Add a python object into the current scope. If it is a function, only that will be added. Otherwise, it and all attributes will be added with name mangling. If the second argument is not provided, the base name will be the same as the object's but pascal-cased.

```py
(Include "str" String)

(StringSplit "a b c")  # => ["a", "b", "c"]
```

### `True`, `False`, `None`, `NotImplemented`

Same singletons as Python

### `Int`, `Float`, `Complex`

Base numeric types

### `List`, `Tuple`, `Set`, `FrozenSet`

Base container types

```py
# (note that commas are also ignored)
(Tuple@ 0, 0)  # => (0, 0)
```

### `Print`, `ReadLine`, `Input`

IO functions. `ReadLine` leaves trailing newlines and does not raise `EOFError`

### `Abs`

Absolute value.

### `Min`, `Max`

Minimum or maximum value in sequence (or all arguments)

### `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`

Arithmetic functions

### `Neg`

Unary negation (`-x`)

### `AndB`, `XorB`, `InvB`, `OrB`, `LShift`, `RShift`

Bitwise functions (`&`, `^`, `~`, `|`, `<<`, `>>`)

### `Not`

Unary logical not

### `Eq`, `Lt`, `Le`, `Neq`, `Gt`, `Ge`

Comparison functions (`==`, `<`, `<=`, `!=`, `>`, `>=`)

### `First`, `Second`, `Third`, `Fourth`, `Fifth`, `Sixth`, `Seventh`, `Eighth`, `Ninth`, `Tenth`, `Last`

Indexing shorthands

### `Nth`

Return the Nth element of a list, can be chained

```py
(Def@ MyArray (List@ 2, 3, 4))

(Nth@ 1 @MyArray)  # => 3

(Def@ MyNestedArray (List (List@ 1, 2, 3) (List@ 4, 5, 6)))

(Nth@ 1, 2 @MyNestedArray)  # => 6
```

### `Length`

Sequence length (`len`)
