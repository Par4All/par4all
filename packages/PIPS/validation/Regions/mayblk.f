      SUBROUTINE MAYBLK(A,J)
      INTEGER A(5), J, I
C
      I = J
      A(I) = A(I-1) + A(I+1)
C
      END
