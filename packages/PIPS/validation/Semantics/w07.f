      PROGRAM W07

      PARAMETER (N=10)
      INTEGER A(N)

      I = 1
      J = N
      DO WHILE (I.LT.N)
         A(I) = A(J)
         J = J-1
         I = I+1
      ENDDO
      PRINT *, 'that is all'
      END
