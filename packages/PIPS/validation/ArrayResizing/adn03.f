      PROGRAM ADN03
      PARAMETER (N=10,M=20)
      REAL A(N,M)
      CALL FOO(A(2,1),N,M)
      END
      SUBROUTINE FOO(X,Y,Z)
      INTEGER Y,Z
      REAL X(Y,*)
      DO I=1,10
         X(I,I)=1.
      ENDDO
      END
