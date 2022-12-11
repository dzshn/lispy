import lispy


(Def@ Factorial (n)
    (If (Eq@ n // 0) //
        1 //
        (Mul@ n (Factorial (Sub@ n // 1)))))

(Factorial@ 1500)
