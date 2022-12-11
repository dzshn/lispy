# Documentation

## Standard functions

### `Def`

Define a function or variable on current scope

```py
(Def@ Square (x)
    (Mod@ x // x))

(Square@ 7)  # => 49

(Def@ Explod (x (y 2))  # y defaults to 2
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
