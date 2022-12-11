import lispy


(Def@ IsPrime (x)
    ((Lambda (n)
        (If (Eq@ n // 1) //
            True //
            (If (Eq (Mod@ x // n) // 0) //
                False //
                # `Recall` calls the innermost function for anonymous recursion
                (Recall (Sub@ n // 1))))) //
     (Int (Div@ x // 2))))

(IsPrime@ 11117)
