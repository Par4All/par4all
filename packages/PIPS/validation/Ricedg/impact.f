      PROGRAM IMPACT
      COMMON /COM/ T
      INTEGER A, B, C(5), D
      A = 0
      B = 1
C     no alias
      CALL FOO(A, B, C, D)
C     alias between two scalars
      CALL FOO(A, A, C, D)
      CALL FOO(A, B, C, A)
C     alias between one formal variable with one common variable
      CALL FOO(T, B, C, D)
C     alias between one scalars and one array
      CALL FOO(A, B, C, C(1))
C     no alias
      CALL FOO3(A, B)
C     alias between two arrays
      CALL FOO1(C(1), C(2))
      END

      SUBROUTINE FOO(X,Y,Z,ZZ)
      COMMON /COM/ T
      INTEGER X, Y, Z(5), ZZ, I
      X = X + 1
      T = Y
      Y = X
      X = T
      I = 1
      DO WHILE (I.LT.5)
         Z(I) = ZZ * Z(I)
      ENDDO
      DO I = 1, Y
         X = X + 1
      ENDDO
      END

      SUBROUTINE FOO1(X,Y)
      INTEGER X(5), Y(4)
      CALL FOO2(X(2), Y(2))
      END

      SUBROUTINE FOO2(X, Y)
      INTEGER X(4), Y(3)
      CALL FOO3(X(2),Y(1))
      END

      SUBROUTINE FOO3(X,Y)
      INTEGER X, Y
      X = 4
      T = T + X
      Y = X
      END



