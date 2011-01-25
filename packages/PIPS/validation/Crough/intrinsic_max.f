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
      I = MAX (I, J, K)
      I = MAX0 (I, J, K)
      F = AMAX1 (F, G, H)
      X = DMAX1 (X, Y, Z)
      I = MAX1 (F, G, H)
      F = AMAX0 (I, J, K)
      END
