# Lispy: Lisp-like Python magic

```py
import lispy

(Def @Factorial (n)
    (If (Eq @n @0)
        @1
        (Mul @n (Factorial (Sub @n @1)))))

(Print (Factorial @10))  # => 3628800
```

