C     Derived from muller01.f

C     Used to debug the translation of OUT effects through a loop
C
C     The print region is A, but under the condition I=11
C
C     If the loop transformer used indicates 1<=I<=10, the application
C     of the reverse transformer leads to an empty OUT region for the loop

      SUBROUTINE SUB(I)
      INTEGER  I
      INTEGER A(10)

      DO I = 1, 10
	A(I)=I
      ENDDO
      PRINT *,A
      END
