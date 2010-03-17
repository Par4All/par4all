      PROGRAM TEST10
      PARAMETER (N=50)
      REAL A(100)
      DO I=1,N,2
            A(I) = A(2*I)
      ENDDO
      STOP
      END
