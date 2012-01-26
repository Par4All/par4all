      PROGRAM PTRAP

      INTEGER A(10)

      CALL TRAP(A, 10)
      CALL TRAP(A, 10)
      
      END


      SUBROUTINE TRAP(B, N)

      INTEGER N, B(N), C

      SAVE C

      DATA C /0/

      C = C + 1

      B(C) =  C

      END
