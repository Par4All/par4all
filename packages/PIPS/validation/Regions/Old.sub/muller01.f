      PROGRAM MAIN
      INTEGER  A(3)
      INTEGER I

      DO I = 1, 3
         A(I) = I
      ENDDO
      PRINT *,A

      CALL SUB(I)
      END


      SUBROUTINE SUB(I)
      INTEGER  I
      INTEGER A(10)

      DO I = 1, 10
	A(I)=I
      ENDDO
      PRINT *,A
      END
