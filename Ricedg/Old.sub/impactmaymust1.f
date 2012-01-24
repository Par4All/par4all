      PROGRAM IMPACT
      COMMON /COM/ T
      INTEGER A, B, C(5), D
      A = 0
      B = 1
C     alias between two scalars
      CALL FOO(A, A, C, D)
      CALL FOO(A, B, C, A)
C     alias between one scalars and one array
      CALL FOO(Z, B, Z, ZZ)
      END

      SUBROUTINE FOO(X,Y,Z,ZZ)
      COMMON /COM/ T
      INTEGER X, Y, Z(5), ZZ, I
      X = X + 1
      T = Y
      I = 6
      DO WHILE (I.LT.5)
         Z(I) = ZZ * Z(I)
      ENDDO
      Y = 2
      DO I = 1, Y
         X = X + 2
      ENDDO
      END

C interval_graph
