      SUBROUTINE PMAT(V,W)
C
      PARAMETER(NFAC=49)
      COMMON/OACOF/AA(NFAC,NFAC)
      DIMENSION V(1),W(1),C(NFAC)
C
      DO 10,I=1,NFAC
      DO 20,J=1,NFAC
   20 C(J)=AA(I,J)
C
      W(I)=SDOT(NFAC,C,1,V,1)
   10 CONTINUE
C
C
      RETURN
      END
       FUNCTION SDOT(N,X,INCX,Y,INCY)
       REAL*4 X(1),Y(1),SDOT
       SDOT=0.0
       IX=1
       IY=1
       DO 10 I=1,N
          SDOT=SDOT+X(IX)*Y(IY)
          IX=IX+INCX
          IY=IY+INCY
  10   CONTINUE
       RETURN
       END
