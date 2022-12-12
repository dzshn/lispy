import lispy

(Include@ "str.split")
(Include@ "set.union")

(Def@ Direction (dir)
    (Cond ((Eq@ dir, "D"), (Tuple@ 1, 0)),
          ((Eq@ dir, "L"), (Tuple@ 0, 1)),
          ((Eq@ dir, "U"), (Tuple@ (Neg@ 1), 0)),
          ((Eq@ dir, "R"), (Tuple@ 0, (Neg@ 1)))))

(Def@ Clamp (x)
    (Int (And@ x (Div@ x (Abs@ x)))))

(Def@ Distance (a, b)
    (Max (Abs (Sub (First@ a) (First@  b)))
         (Abs (Sub (Second@ a) (Second@ b)))))

(Def@ MoveHead (head, dir)
    (Tuple (Add (First@  head) (First@  dir))
           (Add (Second@ head) (Second@ dir))))

(Def@ MoveSegment (tail, head)
    (If (Ge (Distance@ tail, head), 2)
        (Tuple (Add (First@  tail) (Clamp (Sub (First@  head) (First@  tail))))
               (Add (Second@ tail) (Clamp (Sub (Second@ head) (Second@ tail))))),
        tail))

(Def@ MoveSegments (knots, (n* 0))
    (If (Eq@ n, (Sub (Length@ knots), 1)),
        knots,
        (Recur (Add (Take (Add@ n, 1), knots)
                    (List (MoveSegment (Nth (Add@ 1, n), knots) (Nth@ n, knots)))
                    (Drop (Add@ n, 2), knots)),
            (Add@ 1, n))))

(Def@ Move (knots, dir, times, visited)
    (If (Eq@ times, 0)
        (Tuple@ knots, visited)
        (Let (head (MoveHead (Nth@ 0, knots), dir))
            (Let (knots (MoveSegments (Add (List@ head) (Drop@ 1, knots))))
                (Recur@ knots, dir, (Sub@ times, 1), (SetUnion@ visited, (Set (Last@ knots))))))))

(Def@ Aoc ((knots* (Mul@ 10, (List@ (Tuple@ 0, 0)))) (visited@ (Set@ (Tuple@ 0, 0))))
    (Let (line (StrSplit (ReadLine)))
        (If (Not@ line),
            visited,
            (ApplyRecur@ (Move@ knots, (Direction@ (First@ line)), (Int@ (Second@ line)), visited)))))

(Length (Aoc))
