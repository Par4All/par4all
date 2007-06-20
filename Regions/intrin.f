C
      PROGRAM INTRP
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION AA(5)
      DIMENSION BB(5)
C     
      DO I = 1,5
         BB(I) = I
      ENDDO   
C
      CALL INTRIN(AA,BB)
C
      END
C
C
      SUBROUTINE INTRIN(A,B)
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(5)
      DIMENSION B(5)
C
      DO I = 1,5
         A(I) = SIN(B(I))
      ENDDO
C
      END
