import lispy


(Def @IsPrime (n (i @0))
    (Let (i (If (Eq @i @0) (Int (Div @n @2)) @i))
        (If (Eq @i @1)
            @True
            (If (Eq (Mod @n @i) @0)
                @False
                (IsPrime @n (Sub @i @1))))))

(Print (IsPrime @64)) # => False
(Print (IsPrime @127)) # => True
