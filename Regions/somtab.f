C
      PROGRAM SOMP
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION AA(5, 5)
      DIMENSION BB(5,5)
C      

      CALL SOMTAB(AA,BB)
C
      END
C
C     On doit obtenir une region MAY pour A
C     et une region MUST pour B, et ces tags
C     doivent etre conserve's pour les summary regions
C
      SUBROUTINE SOMTAB(A,B)
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(5,5)
      DIMENSION B(5,5)
C
      B(1,1) = 3
      M = A(1,1) + A(3,3)
C
      END
