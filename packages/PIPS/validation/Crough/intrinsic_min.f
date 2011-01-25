      PROGRAM FOO
      INTEGER*4 I, J, K
      REAL*4 F, G, H
      REAL*8 X, Y, Z
      I = 1
      J = 2
      K = 3
      F = 1
      G = 2
      H = 3
      X = 1
      Y = 2
      Z = 3
      I = MIN (I, J, K)
      I = MIN0 (I, J, K)
      F = AMIN1 (F, G, H)
      X = DMIN1 (X, Y, Z)
      I = MIN1 (F, G, H)
      F = AMIN0 (I, J, K)
      END
