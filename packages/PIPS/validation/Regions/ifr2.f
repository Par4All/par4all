C
      PROGRAM IFR2P
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION AA(5)
      DIMENSION BB(5)
C     
      DO I = 1,5
         AA(I) = I
         BB(I) = 0
      ENDDO
         
      CALL IFR2(AA,BB)
C
      END
C
C
      SUBROUTINE IFR2(A,B)
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(5)
      DIMENSION B(5)
C
      DO I = 1,5
         IF (A(I) .GT. 2) THEN
            B(I) = A(I) - 1
            ELSE
               A(I) = B(I) + 1
         ENDIF
      ENDDO
C
      END
