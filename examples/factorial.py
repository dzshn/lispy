import lispy


(Def@ Factorial (n)
    ((Lambda (p, n)
        (If (Eq@ n, 0) //
            p //
            (Recur (Mul@ p, n) (Sub@ n, 1)))) //
     1, n))

(Factorial@ 1500)
