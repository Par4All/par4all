C***********************************************************************
      REAL*8 FUNCTION SDOT(L,X,INCX,Y,INCY)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION X(*),Y(*)
      IX0=1-INCX
      IY0=1-INCY
      SDOT=0.
      DO 1 I=1,L
1     SDOT=SDOT+X(IX0+I*INCY)*Y(IY0+I*INCY)
      RETURN
	end
