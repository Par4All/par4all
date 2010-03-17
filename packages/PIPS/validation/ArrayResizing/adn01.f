      PROGRAM ADN01
      PARAMETER (N=10,M=20)
      REAL A(N,M)
      I = 5     
      CALL FOO(A,N,M)
      J = 9
      END
      SUBROUTINE FOO(X,Y,Z)
      INTEGER Y,Z
      REAL X(Y,*)
      DO I=1,10
         X(I,I)=1.
      ENDDO
      END

