      PROGRAM W02
      INTEGER I, N
      REAL T(10)

      I = 1
      N = 10

      DO WHILE (I.LT.N)
         PRINT *, T(I)
         I = I+1
      ENDDO

      PRINT *, I

      END
