C NN small example about alias 1 caller

      PROGRAM ALIAS
      COMMON /FOO/ X
      EQUIVALENCE (Z,X)
      CALL FOO1(C,C)
      CALL FOO2(Z)
      CALL FOO2(X)      
      END

      SUBROUTINE FOO1(A,B)
      END

      SUBROUTINE FOO2(V)
      COMMON /FOO/ Y
      END

