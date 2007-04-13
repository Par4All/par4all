C     Extract from four2.f

C     No interesting fix-point can be found because the convex hull does
C     not work and because the transformer fix-point would require
C     precondition information.

      SUBROUTINE FOUR3(N)

      J = 1
      NN = 2*N
      M=NN/2
 1    IF ((M.GE.2).AND.(J.GT.M)) THEN
         J=J-M
         M=M/2
         GO TO 1
      ENDIF

      PRINT *, N, NN, M, J

      END
