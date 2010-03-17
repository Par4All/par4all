      SUBROUTINE NON_LINEAR13 (IS, M, U, X, Y, NQ)

      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER (PI = 3.141592653589793238D0)
      COMMON /COUNT/ KTTRANS(256)
      DIMENSION U(1), X(1), Y(1)

      if(x.gt.0) then
         is = -1
      else
         is = 1
      endif

      IF (IS .NE. 1 .AND. IS .NE. -1)
     $  THEN
         print *, is
      ENDIF
      END
