      PROGRAM MAIN
      PARAMETER (N=10)
      REAL A(N)
      CALL FOO(A,N)
      END
      SUBROUTINE FOO(X,Y)
      INTEGER Y
      REAL X(*)
      DO I=1,10
         X(I)=1.
      ENDDO
      END
