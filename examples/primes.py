import lispy


(Def@ IsPrime (n (i // 0))
    (Let (i (If (Eq@ i // 0) // (Int (Div@ n // 2)) // i))
        (If (Eq@ i // 1) //
            True //
            (If (Eq (Mod@ n // i) // 0) //
                False //
                (IsPrime@ n (Sub@ i // 1))))))

(Print@ "64:" //    (IsPrime@ 64))    # => False
(Print@ "127:" //   (IsPrime@ 127))   # => True
(Print@ "9999:" //  (IsPrime@ 9999))  # => False
(Print@ "11117:" // (IsPrime@ 11117)) # => True
