C Dynamic allocation of arrays.
C The size might well be some large number,say, 10000. Once that's allocated,
C the subroutines perform their tasks, not knowing that the array was dynamically allocated

      PROGRAM MAIN
      REAL X
      POINTER (P,X)
      READ *, NSIZE
c      P = MALLOC(NSIZE)
      CALL CALC(X,NSIZE)
      END
      SUBROUTINE CALC(A,N)
      REAL A(N)
      DO I =1,N
         A(I) = I
      ENDDO
      END
