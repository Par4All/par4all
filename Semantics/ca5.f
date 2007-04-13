      PROGRAM CA5

C     Check that read effects in declarations are not forgotten

      INTEGER A(10)

      READ *, N, M

      if(n.ge.m) then
         CALL RUN(A,N,M)
         print *, n, m
      endif
     
      END

      SUBROUTINE RUN(A,N,M)
      INTEGER A(N)

      DO 10 I =1, M
        A(I)=1.
 10   CONTINUE

      END

