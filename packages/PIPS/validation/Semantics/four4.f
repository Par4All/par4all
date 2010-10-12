C     Structured version of four3.f
C
C     The loop here is never entered... since J is initialized to 1

      SUBROUTINE FOUR4(N)

      J = 1
      NN = 2*N
      M=NN/2
      DO WHILE ((M.GE.2).AND.(J.GT.M))
         J=J-M
         M=M/2
      ENDDO

      PRINT *, N, NN, M, J

      END
