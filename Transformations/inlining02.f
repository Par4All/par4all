      SUBROUTINE SWAP(A,B)
      INTEGER A,B,C
      C=A
      A=B
      B=C
      END

      SUBROUTINE BAR
      INTEGER A,B
      A=1
      B=2
      CALL SWAP(A,B)
      PRINT *, A , B
      END
