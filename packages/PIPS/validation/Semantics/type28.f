      program type28

C     Check impact of equivalence on non-integer types

      real x, y, u, v
      logical l, l1, l2, l3
      character*4 s, s1, s2, s3

      equivalence (i,x) , (u, v), (i, l1), (l2, l3), (i, s1), (s2, s3)

      y = x
      y = u
      y = v
      x = 1.0

      l = l1
      l = l2
      l = l3
      l1 = .TRUE.

      s = s1
      s = s2
      s = s3
      s1 = "TOTO"

      end
