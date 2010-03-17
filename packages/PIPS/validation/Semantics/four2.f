C     Modified version of four1.f to study behaviors of unstructured

*****|******************************************************************|
      SUBROUTINE FOUR2(DATA,N,IX)
*****|******************************************************************|
*     Replaces DATA by its discrete Fourier transform
*     IX      : >0 Fourier Analysis
*               <0 Fourier Synthesis
*     DATA    : Complex array of length N.
*               N must be an integer power of 2 (non verifie)
*-----------------------------------------------------------------------|
      INTEGER  N,IX
      REAL DATA(2*N)
*
      INTEGER  I,J,IXX,NN,M,MMAX,ISTEP
      REAL     TEMPI,TEMPR
      DOUBLE PRECISION WR,WI,WPR,WPI,WTEMP,THETA
*-----------------------------------------------------------------------|
      IXX=ISIGN(1,IX)
      NN=2*N
*
*     Bit reversal
*     ------------
      J=1
      DO 11 I=1,NN,2
        IF(J.GT.I)THEN
          TEMPR=DATA(J)
          TEMPI=DATA(J+1)
          DATA(J)=DATA(I)
          DATA(J+1)=DATA(I+1)
          DATA(I)=TEMPR
          DATA(I+1)=TEMPI
        ENDIF
        M=NN/2
1       IF ((M.GE.2).AND.(J.GT.M)) THEN
          J=J-M
          M=M/2
        GO TO 1
        ENDIF
        J=J+M
11    CONTINUE
*
*     Danielson-Lanczos algorithm
*     ---------------------------
      MMAX=2
2     IF (NN.GT.MMAX) THEN
c         IF (NN.GT.MMAX.AND.MMAX.GE.2) THEN
        ISTEP=2*MMAX
        THETA=6.28318530717959D0/(IXX*MMAX)
        WPR=-2.D0*DSIN(0.5D0*THETA)**2
        WPI=DSIN(THETA)
        WR=1.D0
        WI=0.D0
        DO 13 M=1,MMAX,2
          DO 12 I=M,NN,ISTEP
            J=I+MMAX
            TEMPR=SNGL(WR)*DATA(J)-SNGL(WI)*DATA(J+1)
            TEMPI=SNGL(WR)*DATA(J+1)+SNGL(WI)*DATA(J)
            DATA(J)=DATA(I)-TEMPR
            DATA(J+1)=DATA(I+1)-TEMPI
            DATA(I)=DATA(I)+TEMPR
            DATA(I+1)=DATA(I+1)+TEMPI
12        CONTINUE
          WTEMP=WR
          WR=WR*WPR-WI*WPI+WR
          WI=WI*WPR+WTEMP*WPI+WI
13      CONTINUE
        MMAX=ISTEP
c        endif
      GO TO 2
      ENDIF
*
      END
