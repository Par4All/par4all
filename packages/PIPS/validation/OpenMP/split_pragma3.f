      PROGRAM SPLIT_PRAGMA3

      INTEGER N, J
      INTEGER T2, T1, L, JP2, JP1, JM2, JM1, IP2, IP1, IM2, IM1, I
      PARAMETER (N=10)
      REAL A(N)

      DO J = 1, N
         T2  = J + 1
         T1  = T2 + 1
         L   = T1 + 1
         JP2 = L + 1
         JP1 = JP2 + 1
         JM2 = JP1 + 1
         JM1 = JM2 + 1
         IP2 = JM1 + 1
         IP1 = IP2 + 1
         IM2 = IP1 + 1
         IM1 = IM2 + 1
         I   = IM1 + 1
         A(J) = I*1.0
      ENDDO
      PRINT *, A
      END
