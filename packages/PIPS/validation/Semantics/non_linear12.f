      SUBROUTINE NON_LINEAR12 (IS, M, U, X, Y, NQ)

      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER (PI = 3.141592653589793238D0)
      COMMON /COUNT/ KTTRANS(256)
!     X is used as a scalar variable below...
!      DIMENSION U(1), X(1), Y(1)
      DIMENSION U(1), Y(1)

      if(x.gt.0) then
         is = -1
      else
         is = 1
      endif

      if(x.gt.0) then
         m = 5
      else
         m = 6
      endif

      if(x.gt.0) then
         nq = 1
      else
         nq = 64
      endif

      MX = MOD (K, 64)

      IF ((IS .NE. 1 .AND. IS .NE. -1) .OR. M .LT. 1 .OR. M .GT. MX)    
     $  THEN
         PRINT *,  IS, M, MX
      ELSE
         PRINT *,  IS, M, MX
      ENDIF

      print *, mx
      END
