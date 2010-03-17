C
      PROGRAM IFR1P
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION AA(5)
C     
      DO I = 1,5
         AA(I) = I
      ENDDO
         
      CALL IFR1(AA)
C
      END
C
C
      SUBROUTINE IFR1(A)
C
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(5)
C
      DO I = 1,5
         IF (A(I) .GT. 2) THEN
            A(I) = A(I) - 1
            ELSE
               A(I) = A(I) + 1
         ENDIF
      ENDDO
C
      END
